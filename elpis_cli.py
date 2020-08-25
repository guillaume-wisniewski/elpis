import sys
import argparse
import json

from elpis.engines import Interface, ENGINES
from elpis.engines.kaldi.errors import KaldiError


def list_datasets(args):

    interface = Interface.load(args.working_dir)
    
    print("\n\n")
    print("Name\t\tDirectory")
    print("----\t\t---------")

    for name in interface.list_datasets():
        hash_dir = interface.config['datasets'][name]
        path = interface.datasets_path.joinpath(hash_dir)
        print(f"{name}\t\t{path}")

def delete_dataset(args):
    interface = Interface.load(args.working_dir)
    interface.delete_dataset(args.dsname)


def create_dataset(args):

    interface = Interface.load(args.working_dir)
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
    
    ds.validate()
    ds.process()
        

def use_existing_split(corpus_filename, train_filename, test_filename):
    print("youpi")

    for train_ex in json.load(open(train_filename)):
        yield "TRAIN", train_ex

    for test_ex in json.load(open(test_filename)):
        yield "TEST", test_ex

    
def train(args):
    from functools import partial
    
    interface = Interface.load(args.working_dir)

    ds = interface.get_dataset(args.dsname)
    
    interface.set_engine(ENGINES["espnet"])
    m = interface.new_model(args.model_name)
    m.link(ds, None)
    m.build_structure(split_train_test=partial(use_existing_split, train_filename=args.train, test_filename=args.test))

    def finish():
        print("victory!!!")

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
    train_model_parser.add_argument("--train", required=True)
    train_model_parser.add_argument("--test", required=True)
    train_model_parser.set_defaults(func=train)
    
    args = parser.parse_args()
    args.func(args)
