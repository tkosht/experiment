from collections import namedtuple

ProjectRecord = namedtuple(
    "ProjectRecord",
    [
        "box_idx",
        "img_url",
        "page",
        "detail_url",
        "area",
        "title",
        "meter",
        "category",
        "owner",
        "current_funding",
        "supporters",
        "remaining_days",
        "status",
    ],
)


ProjectDetails = namedtuple(
    "ProjectDetails",
    [
        "type",
        "detail_url",
        "title",
        "img_url",
        "backer_amount",
        "abstract",
        "article_text",
        "article_html",
        "profile_text",
        "profile_url",
        "icon_url",
        "user_name",
        "prefecture",
        "project_exprience",
        "readmore",
        "return_boxes",
    ],
)

ReturnBox = namedtuple("ReturnBox", ["return_idx", "return_img_url", "price", "desc"])
