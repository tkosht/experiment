from fastapi import Body, FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    point: str
    params: dict = {}


@app.post("/api/dify/receive")
async def dify_receive(data: InputData = Body(...), authorization: str = Header(None)):
    """
    DifyからのAPIクエリデータを受信します。
    """
    expected_api_key = "123456"  # TODO このAPIのAPIキー
    try:
        auth_scheme, _, api_key = authorization.partition(" ")

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized {e}")

    if auth_scheme.lower() != "bearer" or api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    point = data.point

    # デバッグ用
    print(f"point: {point}")

    if point == "ping":
        return {"result": "pong"}
    if point == "app.external_data_tool.query":
        return handle_app_external_data_tool_query(params=data.params)
    # elif point == "{point name}":
    # TODO その他のポイントの実装

    raise HTTPException(status_code=400, detail="Not implemented")


def handle_app_external_data_tool_query(params: dict):
    app_id = params.get("app_id")
    tool_variable = params.get("tool_variable")
    inputs = params.get("inputs")
    query = params.get("query")

    # デバッグ用
    print(f"app_id: {app_id}")
    print(f"tool_variable: {tool_variable}")
    print(f"inputs: {inputs}")
    print(f"query: {query}")

    # TODO 外部データツールクエリの実装
    # 返り値は"result"キーを持つ辞書でなければならず、その値はクエリの結果でなければならない
    if inputs.get("location") == "London":
        return {
            "result": "City: London\nTemperature: 10°C\nRealFeel®: 8°C\nAir Quality: Poor\nWind Direction: ENE\nWind "
            "Speed: 8 km/h\nWind Gusts: 14 km/h\nPrecipitation: Light rain"
        }
    else:
        return {"result": "Unknown city"}
