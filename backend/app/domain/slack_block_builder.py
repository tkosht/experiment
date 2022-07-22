from app.component.params import add_args


@add_args(params_file="conf/app.yml", root_key = "/slack/client/replying_server", as_default = False)
def build_image_url(image_id: str, base_url: str) -> str:
    image_url: str = f"{base_url}/image?image_id={image_id}"
    return image_url


def build_block_image(image_id: str, msg: str) -> dict:
    image_url: str = build_image_url(image_id)
    block: dict = {
            "type": "image",
            "title": {
                "type": "plain_text",
                "text": "Please enjoy this wordcloud"
            },
            "block_id": "image4",
            "image_url": image_url,
            "alt_text": msg
        }
    return block
