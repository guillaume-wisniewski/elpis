from ..blueprint import Blueprint
from flask import current_app as app, jsonify, request
from elpis.engines import Interface, ENGINES
import shutil
import os
import glob

bp = Blueprint("config", __name__, url_prefix="/config")


@bp.route("/reset", methods=['GET', 'POST'])
def reset():
    current_interface_path = app.config['INTERFACE'].path
    app.config['INTERFACE'] = Interface(current_interface_path)
    data = {
        "message": "reset ok"
    }
    return jsonify({
        "status": 200,
        "data": data
    })


@bp.route("/engine/list", methods=['GET', 'POST'])
def engine_list():
    data = {
        'engine_list': list(ENGINES.keys())
    }
    return jsonify({
        "status": 200,
        "data": data
    })


@bp.route("/engine/load", methods=['GET', 'POST'])
def engine_load():
    engine_name = request.json["engine_name"]
    if engine_name not in ENGINES:
        return jsonify({"status": 404,
                        "data": "Engine not found in ENGINES"})
    engine = ENGINES[engine_name]
    interface = app.config['INTERFACE']
    interface.set_engine(engine)
    data = {
        "engine": engine_name
    }
    return jsonify({
        "status": 200,
        "data": data
    })


@bp.route("/object-names", methods=['GET', 'POST'])
def object_names():
    interface: Interface = app.config['INTERFACE']
    data = {
        "object_names": {
            "datasets": interface.list_datasets(),
            "pron_dicts": interface.list_pron_dicts_verbose(),  # includes pd name and ds name
            "models": interface.list_models()
        }
    }
    print(data)
    return jsonify({
        "status": 200,
        "data": data
    })

