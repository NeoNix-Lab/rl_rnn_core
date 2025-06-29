import json
from typing import Optional
from .base_models import BaseModelsClass

class RewardFunction(BaseModelsClass):
    table_name: str = "functions"
    fields = ["name", "function", "data_schema", "action_schema", "status_schema", "notes"]

    DB_SCHEMA = """
    CREATE TABLE IF NOT EXISTS functions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        function TEXT,
        data_schema TEXT,
        action_schema TEXT,
        status_schema TEXT,
        notes TEXT
    );
    """

    INSERT_QUERY = """
    INSERT INTO functions (name, function, data_schema, action_schema, status_schema, notes)
    VALUES (?, ?, ?, ?, ?, ?);
    """

    def __init__(self,
                 name: str,
                 function: str,
                 data_schema: dict,
                 action_schema: dict,
                 status_schema: dict,
                 notes: str = '',
                 id: Optional[int] = None):
        super().__init__(id=id, name=name, function=function,
                         data_schema=json.dumps(data_schema),
                         action_schema=json.dumps(action_schema),
                         status_schema=json.dumps(status_schema),
                         notes=notes)
        self._decoded = {
            "data_schema": data_schema,
            "action_schema": action_schema,
            "status_schema": status_schema
        }

    def push_on_db(self) -> None:
        super().push_on_db()  # crea tabella e inserisce

    @classmethod
    def convert_db_response(cls, row) -> "RewardFunction":
        inst = super().convert_db_response(row)
        # decodifica JSON
        inst._decoded = {
            "data_schema": json.loads(inst.data_schema),
            "action_schema": json.loads(inst.action_schema),
            "status_schema": json.loads(inst.status_schema)
        }
        return inst


