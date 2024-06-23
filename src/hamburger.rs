use leptos::{component, ev::MouseEvent, spawn_local, view, IntoView};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::to_value;

use crate::invoke;

#[derive(Serialize, Deserialize)]
struct OpenDevToolsArgs {}

#[component]
pub fn Hamburger() -> impl IntoView {
    let open_devtools = move |ev: MouseEvent| {
        ev.prevent_default();
        spawn_local(async move {
            println!("????");
            let args = to_value(&OpenDevToolsArgs {}).unwrap();
            invoke("toggle_devtools", args).await.as_string().unwrap();
        });
    };

    let dev_tools_option = "🔧 Toggle Dev Tools";
    view! {
        <div class="hamburger-menu">
            <input id="menu__toggle" type="checkbox" />
            <label class="menu__btn" for="menu__toggle">
                <span></span>
            </label>
            <ul class="menu__box">
                <li><a class="menu__item" href="#" on:click=open_devtools>{dev_tools_option}</a></li>
                <li><a class="menu__item" href="#">About</a></li>
                <li><a class="menu__item" href="#">Team</a></li>
                <li><a class="menu__item" href="#">Contact</a></li>
                <li><a class="menu__item" href="#">Twitter</a></li>
            </ul>
        </div>
    }
}
