use super::hamburger::Hamburger;
use ev::MouseEvent;
use leptos::leptos_dom::ev::SubmitEvent;
use leptos::*;
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::to_value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"])]
    pub async fn invoke(cmd: &str, args: JsValue) -> JsValue;
}

#[derive(Serialize, Deserialize)]
struct GreetArgs<'a> {
    name: &'a str,
}

#[derive(Serialize, Deserialize)]
struct PlopArgs {}

#[component]
pub fn App() -> impl IntoView {
    let (name, set_name) = create_signal(String::new());
    let (greet_msg, set_greet_msg) = create_signal(String::new());
    let (image_url, set_image_url) = create_signal::<Option<String>>(None);

    let update_name = move |ev| {
        let v = event_target_value(&ev);
        set_name.set(v);
    };

    let greet = move |ev: SubmitEvent| {
        ev.prevent_default();
        spawn_local(async move {
            let name = name.get_untracked();
            if name.is_empty() {
                return;
            }

            let args = to_value(&GreetArgs { name: &name }).unwrap();
            // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
            let new_msg = invoke("greet", args).await.as_string().unwrap();
            set_greet_msg.set(new_msg);
        });
    };

    let process_image = move |ev: MouseEvent| {
        ev.prevent_default();
        spawn_local(async move {
            println!("????");
            let args = to_value(&PlopArgs {}).unwrap();
            let process_msg = invoke("plop", args).await.as_string().unwrap();
            set_image_url.set(Some(process_msg));
        });
    };

    let image_view = move || match image_url.get() {
        Some(url) => view! {<p>hi<img src=url></img> </p> }.into_view(),
        None => view! {}.into_view(),
    };

    view! {
        <Hamburger />

        <main class="container">
            <p>"Click on the Tauri and Leptos logos to learn more."</p>

            <form class="row" on:submit=greet>
                <input
                    id="greet-input"
                    placeholder="Enter a name..."
                    on:input=update_name
                />
                <button type="submit">"Greet"</button>
            </form>

            <p><b>{ move || greet_msg.get() }</b></p>

            <button on:click=process_image>Process</button>

            img
            {image_view}


            <div class="viewer">
                <img src="hello.jpg" />
            </div>
        </main>
    }
}
