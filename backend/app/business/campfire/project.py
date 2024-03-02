from collections import namedtuple

ProjectRecord = namedtuple(
    "ProjectRecord",
    [
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
        "detail_url",
        "description",
        "goal_amount",
        "start_date",
        "end_date",
        "location",
        "category",
        "tags",
    ],
)
