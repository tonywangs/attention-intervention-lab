"""Separate CLI keeps the historically hashed command implementation intact."""
import argparse
import sys
import zipfile
from .artifacts import Cancelled


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline zero versus mean ablation comparison')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','evaluate','replay'):
        p=sub.add_parser(name)
        p.add_argument('--root',required=True)
        p.add_argument('--protocol',required=True)
        p.add_argument('--output',required=True)
        if name!='prepare':
            p.add_argument('--data',required=True)
            p.add_argument('--max-seconds',type=float,default=7200)
        if name=='replay':p.add_argument('--reference',required=True)
    p=sub.add_parser('report');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        from .mean_experiment import prepare,evaluate,replay
        if args.command=='prepare':prepare(args.root,args.protocol,args.output)
        elif args.command=='evaluate':evaluate(args.root,args.protocol,args.data,args.output,args.max_seconds)
        elif args.command=='replay':replay(args.root,args.protocol,args.data,args.reference,args.output,args.max_seconds)
        else:
            from .artifacts import transaction,verify_manifest,manifest,write_json
            from .mean_report import render_directory
            verify_manifest(args.input)
            with transaction(args.output) as out:
                render_directory(args.input,out/'report.html')
                write_json(out/'manifest.json',manifest(out))
    except (ValueError,OSError,KeyError,TypeError,TimeoutError,Cancelled,zipfile.BadZipFile,EOFError,IndexError) as exc:
        print(f'attention-mean: {exc}',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
