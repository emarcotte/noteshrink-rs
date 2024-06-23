use std::{cmp::Ordering, collections::HashMap};

use clap::Parser;
use image::Rgb;
use rand::prelude::*;

/// A program for converting pictures into possibly more legible versions of themselves.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File names to process.
    input_files: Vec<String>,

    /// Background value threshold percentage (defaults 0.25 (25%))
    #[arg(long, short, default_value_t = 0.25)]
    value_threshold: f32,

    /// Background saturation threshold percentage (defaults to 0.20 (20%))
    #[arg(long, default_value_t = 0.20)]
    saturation_threshold: f32,

    /// Number of output colors
    #[arg(long, short, default_value_t = 8)]
    num_colors: u8,

    /// Percent of pixels to sample
    #[arg(long, default_value_t = 5)]
    sample_size: u8,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Px8RGB(u8, u8, u8);

// TODO: Do we need this type?
#[derive(Hash, Debug, PartialEq, Eq, Clone)]
struct PxPacked(u32);

impl From<&Px8RGB> for PxPacked {
    fn from(value: &Px8RGB) -> Self {
        PxPacked((value.0 as u32) << 16 | (value.1 as u32) << 8 | value.2 as u32)
    }
}

impl From<u32> for Px8RGB {
    fn from(packed: u32) -> Self {
        Self(
            ((packed >> 16) & 0xff) as u8,
            ((packed >> 8) & 0xff) as u8,
            ((packed) & 0xff) as u8,
        )
    }
}

impl From<&PxPacked> for Px8RGB {
    fn from(value: &PxPacked) -> Self {
        Px8RGB::from(value.0)
    }
}

impl From<&Px8RGB> for image::Rgb<u8> {
    fn from(value: &Px8RGB) -> Self {
        Self([value.0, value.1, value.2])
    }
}

struct SatVal {
    sat: f32,
    val: f32,
}

impl Px8RGB {
    /// Reduces the number of bits per channel by the given amounts. Probably you want to use
    /// [`quantize`] though.
    fn quantize(&self, shift: u8, halfbin: u8) -> Self {
        Px8RGB(
            ((self.0 >> shift) << shift) + halfbin,
            ((self.1 >> shift) << shift) + halfbin,
            ((self.2 >> shift) << shift) + halfbin,
        )
    }

    fn dist(&self, other: &Px8RGB) -> f32 {
        let r = (other.0 as f32) - (self.0 as f32);
        let g = (other.1 as f32) - (self.1 as f32);
        let b = (other.2 as f32) - (self.2 as f32);
        ((r * r) + (g * g) + (b * b)).sqrt().abs()
    }

    fn to_sv(self) -> SatVal {
        let cmax = self.0.max(self.1).max(self.2);
        let cmin = self.0.min(self.1).min(self.2);
        let delta = cmax - cmin;
        let sat = if delta > 0 {
            delta as f32 / cmax as f32
        } else {
            0f32
        };
        let val = cmax as f32 / 255.0;
        SatVal { sat, val }
    }
}

/// Reduces the number of bits per channel in the given image.
fn quantize(image: &[Px8RGB], bits_per_channel: Option<u8>) -> impl Iterator<Item = Px8RGB> + '_ {
    let bits_per_channel = bits_per_channel.unwrap_or(6);

    let shift = 8 - bits_per_channel;
    let halfbin = (1 << shift) >> 1;

    image.iter().map(move |p| p.quantize(shift, halfbin))
}

fn sample_pixels<T, R: Rng + ?Sized>(
    rng: &mut R,
    cli: &Args,
    img: &[Px8RGB],
    cb: impl Fn(&mut R, &mut [Px8RGB]) -> T,
) -> T {
    let total_pixels = img.len();
    let sampled_pixels = usize::max(
        1,
        ((total_pixels as f32) * (cli.sample_size as f32 / 100.)) as usize,
    );

    let mut px = Vec::<Px8RGB>::from(img);
    let (sampled, _) = px.partial_shuffle(rng, sampled_pixels);
    cb(rng, sampled)
}

/// Obtains the background color from an image or array of RGB colors
/// by grouping similar colors into bins and finding the most frequent
/// one.
fn get_bg_color(image: &[Px8RGB], bits_per_channel: Option<u8>) -> Px8RGB {
    let mut frequencies = HashMap::<PxPacked, usize>::new();

    // Convert RGB separated pixels to quantized packed pixels so they're easier to cmp.
    let quantized = quantize(image, bits_per_channel).map(|p| PxPacked::from(&p));

    // Count the unique pixel colors
    for p in quantized {
        *frequencies.entry(p).or_default() += 1;
    }

    // Find the most frequent pixel color
    frequencies
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k)
        .unwrap()
        .into()
}

/// Determine whether each pixel in a set of samples is foreground by comparing it to the
/// background color. A pixel is classified as a foreground pixel if either its value or saturation
/// differs from the background by a threshold.
fn get_fg_mask(
    bg_color: &Px8RGB,
    samples: &[Px8RGB],
    value_threshold: f32,
    saturation_threshold: f32,
) -> Vec<bool> {
    let bg_sat = bg_color.to_sv();
    samples
        .iter()
        .map(|p| p.to_sv())
        .map(|sv| SatVal {
            sat: (bg_sat.sat - sv.sat).abs(),
            val: (bg_sat.val - sv.val).abs(),
        })
        .map(|diff| diff.val >= value_threshold || diff.sat >= saturation_threshold)
        .collect()
}

fn kmeans_recalculate(means: &mut [Px8RGB], px: &[Px8RGB], assignments: &[usize]) {
    for (mean_idx, mean) in means.iter_mut().enumerate() {
        let assigned: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(
                |(px, group)| {
                    if *group == mean_idx {
                        Some(px)
                    } else {
                        None
                    }
                },
            )
            .collect();

        if !assignments.is_empty() {
            let (r, g, b) = assigned.iter().fold((0, 0, 0), |acc, p| {
                (
                    (acc.0 + px[*p].0 as u32),
                    (acc.1 + px[*p].1 as u32),
                    (acc.2 + px[*p].2 as u32),
                )
            });

            mean.0 = (r as f32 / assigned.len() as f32).round() as u8;
            mean.1 = (g as f32 / assigned.len() as f32).round() as u8;
            mean.2 = (b as f32 / assigned.len() as f32).round() as u8;
        }
    }
}

/// This is probably the worst possible implementation of a clustering algorithm one could imagine.
fn kmeans<R: Rng + ?Sized>(
    rng: &mut R,
    px: &[Px8RGB],
    kmeans: usize,
    max_iters: u32,
) -> Vec<Px8RGB> {
    let mut means: Vec<Px8RGB> = px.choose_multiple(rng, kmeans).cloned().collect();
    let mut assignments: Vec<usize> = px.iter().map(|_| 0).collect();

    let mut max_iters = max_iters;

    while max_iters > 0 {
        let mut changed = false;
        for (px_idx, px) in px.iter().enumerate() {
            let closest_group = closest_match(px, &means);

            if assignments[px_idx] != closest_group {
                changed = true;
                assignments[px_idx] = closest_group;
            }
        }

        // If clusters didn't change, bail out.
        if !changed {
            break;
        }

        // otherwise recalculate centers and re-run, maybe.
        max_iters -= 1;

        kmeans_recalculate(&mut means, px, &assignments);
    }
    means
}

/// Extract the palette for the set of sampled RGB values. The first
/// palette entry is always the background color; the rest are determined
/// from foreground pixels by running K-means clustering. Returns the
/// palette
fn get_palette(cli: &Args, img: &[Px8RGB], kmeans_iter: u8) -> Vec<Px8RGB> {
    let value_threshold = cli.value_threshold;
    let saturation_threshold = cli.saturation_threshold;

    let mut rng = rand::thread_rng();

    sample_pixels(&mut rng, cli, img, move |rng, sample| {
        let sample = Vec::from(sample);
        let bg_color = get_bg_color(&sample, Some(6));
        let fg_mask = get_fg_mask(&bg_color, &sample, value_threshold, saturation_threshold);

        let fg_px: Vec<Px8RGB> = sample
            .iter()
            .enumerate()
            .filter_map(|(idx, px)| fg_mask[idx].then_some(px))
            .cloned()
            .collect();
        let mut means = kmeans(rng, &fg_px, (cli.num_colors - 1).into(), kmeans_iter.into());

        means.push(bg_color);
        let len = means.len() - 1;
        means.swap(0, len);
        means
    })
}

fn apply_pallet(img: &[Px8RGB], palette: &[Px8RGB], cli: &Args) -> Vec<Px8RGB> {
    let value_threshold = cli.value_threshold;
    let saturation_threshold = cli.saturation_threshold;

    let bg_color = &palette[0];
    let fg_mask = get_fg_mask(bg_color, img, value_threshold, saturation_threshold);

    // TODO: Change bg color to white via option?
    let mut new_image = vec![*bg_color; img.len()];

    for (fg_idx, is_fg) in fg_mask.iter().enumerate() {
        if *is_fg {
            new_image[fg_idx] = palette[closest_match(&img[fg_idx], palette)];
        }
    }

    new_image
}

fn closest_match(p: &Px8RGB, palette: &[Px8RGB]) -> usize {
    palette
        .iter()
        .map(|color| p.dist(color))
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less))
        .unwrap()
        .0
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Args::parse();

    for f in cli.input_files.iter() {
        let image = image::open(f)?;
        let width = image.width();
        let height = image.height();
        let image = image.into_rgb8();

        let px: Vec<Px8RGB> = image
            .enumerate_pixels()
            .map(|p| Px8RGB(p.2 .0[0], p.2 .0[1], p.2 .0[2]))
            .collect();
        let pallet = get_palette(&cli, &px, 8);

        let reduced = apply_pallet(&px, &pallet, &cli);

        let mut imgbuf = image::ImageBuffer::new(width, height);
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            *pixel = Rgb::<u8>::from(&reduced[(y * width + x) as usize]);
        }
        imgbuf.save("asdf.png")?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use clap::Parser;
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::{quantize, sample_pixels, Px8RGB, PxPacked};

    #[test]
    fn px8rgb_from_u8() {
        assert_eq!(Px8RGB::from(0xff010203), Px8RGB(0x01, 0x02, 0x03));
    }

    #[test]
    fn quantize_px() {
        assert_eq!(
            Px8RGB(253, 252, 254).quantize(2, 2),
            Px8RGB(0xFE, 0xFE, 0xFE)
        );

        assert_eq!(Px8RGB(243, 252, 154).quantize(4, 8), Px8RGB(248, 248, 152));
    }

    #[test]
    fn quantize_vec() {
        assert_eq!(
            quantize(&[Px8RGB(253, 252, 254), Px8RGB(243, 252, 154)], Some(4))
                .collect::<Vec<Px8RGB>>(),
            vec!(Px8RGB(248, 248, 248), Px8RGB(248, 248, 152)),
        );
    }

    #[test]
    fn get_bg_color() {
        assert_eq!(
            super::get_bg_color(
                &[
                    Px8RGB(253, 252, 254),
                    Px8RGB(243, 252, 154),
                    Px8RGB(243, 252, 154)
                ],
                Some(4),
            ),
            Px8RGB(248, 248, 152),
        );
    }

    #[test]
    fn sample_px() {
        let img = vec![Px8RGB(255, 0, 0), Px8RGB(0, 255, 0)];

        let mut args = crate::Args::parse_from(vec![""]);
        args.sample_size = 100;

        let expected = [Px8RGB(255, 0, 0), Px8RGB(0, 255, 0)];

        // Test full sample
        let mut rng = rand::thread_rng();
        let sampled = sample_pixels(&mut rng, &args, &img, |_, actual| actual.to_vec());
        assert_eq!(sampled.len(), 2);
        for e in &expected {
            assert!(sampled.contains(e), "Missing {:?}, have {:?}", e, sampled);
        }

        // Test partial sample (at least 1 px)
        args.sample_size = 5;
        let sampled = sample_pixels(&mut rng, &args, &img, |_, actual| actual.to_vec());
        assert_eq!(sampled.len(), 1);

        args.sample_size = 50;
        let img = vec![
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
            Px8RGB(255, 0, 0),
        ];
        let mut rng = rand::thread_rng();
        let sampled = sample_pixels(&mut rng, &args, &img, |_, actual| actual.to_vec());
        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn pack_from_rgb() {
        assert_eq!(
            PxPacked::from(&Px8RGB(0xff, 0xee, 0xdd)),
            PxPacked(0x00ffeedd)
        );
    }

    #[test]
    fn px8rgb_to_sv() {
        let sat = Px8RGB(255, 155, 55).to_sv();
        assert_abs_diff_eq!(0.78431374, sat.sat);
        assert_abs_diff_eq!(1.0, sat.val);

        let sat = Px8RGB(155, 55, 5).to_sv();
        assert_abs_diff_eq!(0.9677419, sat.sat);
        assert_abs_diff_eq!(0.60784316, sat.val);

        let sat = Px8RGB(0, 0, 0).to_sv();
        assert_abs_diff_eq!(0.0, sat.sat);
        assert_abs_diff_eq!(0.0, sat.val);
    }

    #[test]
    fn get_fg_mask() {
        let bg = Px8RGB(255, 255, 255);
        let img = vec![
            Px8RGB(255, 255, 255),
            Px8RGB(0, 0, 0),
            Px8RGB(0, 0, 255),
            Px8RGB(255, 254, 250),
        ];
        let mask = super::get_fg_mask(&bg, &img, 0.2, 0.25);
        assert_eq!(mask, vec!(false, true, true, false));
    }

    #[test]
    fn kmeans() {
        let px = vec![
            Px8RGB(101, 51, 0),
            Px8RGB(109, 71, 0),
            Px8RGB(110, 63, 0),
            Px8RGB(112, 46, 0),
            Px8RGB(117, 59, 0),
            Px8RGB(122, 37, 0),
            Px8RGB(125, 31, 0),
            Px8RGB(125, 47, 0),
            Px8RGB(131, 188, 0),
            Px8RGB(132, 185, 0),
            Px8RGB(134, 189, 0),
            Px8RGB(137, 221, 0),
            Px8RGB(138, 227, 0),
            Px8RGB(139, 189, 0),
            Px8RGB(140, 200, 0),
            Px8RGB(140, 209, 0),
            Px8RGB(141, 206, 0),
            Px8RGB(141, 213, 0),
            Px8RGB(141, 221, 0),
            Px8RGB(142, 192, 0),
            Px8RGB(143, 182, 0),
            Px8RGB(145, 201, 0),
            Px8RGB(150, 195, 0),
            Px8RGB(153, 234, 0),
            Px8RGB(155, 243, 0),
            Px8RGB(160, 207, 0),
            Px8RGB(160, 221, 0),
            Px8RGB(160, 234, 0),
            Px8RGB(161, 210, 0),
            Px8RGB(161, 219, 0),
            Px8RGB(161, 248, 0),
            Px8RGB(162, 230, 0),
            Px8RGB(165, 207, 0),
            Px8RGB(167, 178, 0),
            Px8RGB(167, 184, 0),
            Px8RGB(168, 239, 0),
            Px8RGB(170, 220, 0),
            Px8RGB(171, 216, 0),
            Px8RGB(172, 185, 0),
            Px8RGB(174, 179, 0),
            Px8RGB(175, 208, 0),
            Px8RGB(176, 225, 0),
            Px8RGB(176, 238, 0),
            Px8RGB(183, 230, 0),
            Px8RGB(189, 238, 0),
            Px8RGB(193, 239, 0),
            Px8RGB(197, 214, 0),
            Px8RGB(199, 254, 0),
            Px8RGB(79, 40, 0),
            Px8RGB(82, 50, 0),
            Px8RGB(83, 62, 0),
            Px8RGB(83, 64, 0),
            Px8RGB(86, 30, 0),
            Px8RGB(86, 62, 0),
            Px8RGB(89, 38, 0),
            Px8RGB(89, 47, 0),
            Px8RGB(90, 62, 0),
            Px8RGB(91, 40, 0),
            Px8RGB(92, 57, 0),
            Px8RGB(96, 63, 0),
        ];

        let mut rng = SmallRng::seed_from_u64(1234);
        let means = super::kmeans(&mut rng, &px, 3, 8);
        assert_eq!(
            means,
            vec!(Px8RGB(98, 51, 0), Px8RGB(148, 199, 0), Px8RGB(173, 231, 0),)
        );
    }
}
