import sys
import argparse
import json

from random import shuffle
from pathlib import Path

from elpis.engines import Interface, ENGINES
from elpis.engines.kaldi.errors import KaldiError

import pandas as pd


def show_models(interface, args):
    interface.set_engine(ENGINES["espnet"])
    for model_name in interface.list_models():
        model = interface.get_model(model_name)
        try:
            per = model.get_train_results()["per"]
            fn = model.get_predictions_fn()
        except:
            per = None
            fn = None

        print(f"{model_name}\t\t\t{per}\t\t\t{fn}")


def list_datasets(interface, args):
    
    print("\n\n")
    name = "Name"
    length = "Length"
    directory = "Directory"
    print(f"{name:<30}\t\t{length:<30}\t\t{directory}")
    print(f"{'-' * len(name):<30}\t\t{'-' * len(length):<30}\t\t{'-' * len(directory)}")

    for name in interface.list_datasets():
        hash_dir = interface.config['datasets'][name]
        path = interface.datasets_path.joinpath(hash_dir)

        annotations = json.load(open(path / "annotations.json"))
        length = sum(a["stop_ms"] - a["start_ms"] for a in annotations)
        length = pd.to_timedelta(length, unit='ms')
        
        print(f"{name:<30}\t\t{str(length).split('.')[0]:<15}\t\t{path}")


def delete_dataset(interface, args):
    interface.delete_dataset(args.dsname)


def create_dataset(interface, args):

    try:
        ds = interface.new_dataset(args.dsname)
    except KaldiError as e:
        if args.no_overriding:
            print(e)
            sys.exit(1)
        interface.delete_dataset(args.dsname)
        ds = interface.new_dataset(args.dsname)
        
    ds.add_directory(args.input_dir)
    ds.select_importer("Elan")
    
    importer = ds.importer
    if args.tier_type is not None:
        importer.set_setting("tier_type", args.tier_type)
    if args.tier_type is not None:
        importer.set_setting("tier_name", args.tier_name)

    assert ds.validate() is None
    ds.process()
        

def use_existing_split(corpus_filename, train_filename, test_filename):
    print("youpi")

    for train_ex in json.load(open(train_filename)):
        yield "TRAIN", train_ex

    for test_ex in json.load(open(test_filename)):
        yield "TEST", test_ex


def split_by_time(corpus_filename, train_time, test_time):

    # convert time from mn into ms
    train_time = train_time * 60 * 1_000
    test_time = test_time * 60 * 1_000
    
    try:
        with open(corpus_filename, "r") as input_file:
            json_transcripts: str = json.loads(input_file.read())
    except FileNotFoundError:
        raise Exception(f"JSON file could not be found: {input_json}")

    shuffle(json_transcripts)

    actual_train_time = 0
    actual_test_time = 0
    total_time = 0
    for ex in json_transcripts:

        total_time += ex["stop_ms"] - ex["start_ms"]
        if total_time < test_time:
            yield "TEST", ex
            actual_test_time += ex["stop_ms"] - ex["start_ms"]
        elif total_time > test_time and total_time < test_time + train_time:
            yield "TRAIN", ex
            actual_train_time += ex["stop_ms"] - ex["start_ms"]

    print(f"train_time = {actual_train_time / 60 / 1_000}")
    print(f"test_time = {actual_test_time / 60 / 1_000}")


def train(interface, args):
    from functools import partial

    assert (args.train_filename and args.test_filename) != (args.train_time and args.test_time), "You should specify either filename or length to specify the train/test split"
    
    ds = interface.get_dataset(args.dsname)

    interface.set_engine(ENGINES["espnet"])
    m = interface.new_model(args.model_name)
    m.link(ds, None)

    if args.train_filename is not None:
        m.build_structure(split_train_test=partial(use_existing_split, train_filename=args.train_filename, test_filename=args.test_filename))
    else:
        m.build_structure(split_train_test=partial(split_by_time, train_time=args.train_time, test_time=args.test_time))
        
    m.train(lambda: print("victory"))

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", default="./models")
    
    subparsers = parser.add_subparsers(help='sub-command help')

    create_ds_parser = subparsers.add_parser("create_dataset", help="create and manipulate datasets")
    create_ds_parser.add_argument("--no_overriding", action="store_true", default=False)
    create_ds_parser.add_argument("--dsname", required=True)
    create_ds_parser.add_argument("--input_dir", required=True)
    create_ds_parser.add_argument("--tier_type", default=None)
    create_ds_parser.add_argument("--tier_name", default=None)
    create_ds_parser.add_argument("--speaker", default=None)
    create_ds_parser.set_defaults(func=create_dataset)

    delete_ds_parser = subparsers.add_parser("delete_dataset", help="delete a dataset")
    delete_ds_parser.add_argument("--dsname", required=True)
    delete_ds_parser.set_defaults(func=delete_dataset)

    list_dataset_parser = subparsers.add_parser("list_dataset", help="create and manipulate datasets")
    list_dataset_parser.set_defaults(func=list_datasets)

    train_model_parser = subparsers.add_parser("train", help="train a model")
    train_model_parser.add_argument("--dsname", required=True)
    train_model_parser.add_argument("--model_name", required=True)
    train_model_parser.add_argument("--train_filename", required=False)
    train_model_parser.add_argument("--test_filename", required=False)
    train_model_parser.add_argument("--train_time", type=int, required=False)
    train_model_parser.add_argument("--test_time", type=int, required=False)
    train_model_parser.set_defaults(func=train)

    show_model_parser = subparsers.add_parser("show_models")
    show_model_parser.set_defaults(func=show_models)
    
    args = parser.parse_args()

    interface_fn = Path(args.working_dir) / "interface.json"
    if interface_fn.is_file():
        interface = Interface.load(args.working_dir)
    else:
        interface = Interface(args.working_dir)

    args.func(interface, args)
