#!/bin/bash
set -e

docs="Usage: \
\n\tbash $0 [options] \
\nOptions: \
\n\t-c cuda: cuda device, required\
\n\t-s seed: random seed, required\
\n\t-r skip_train, optional\
\n\t-t skip_test, optional\
\nExample: \
\n\t bash $0 -c 1 -e"

usage() {
    echo -e $docs >&2
    exit 1
}

if [ $# -eq 0 ] || [ $1 == -h ]; then usage; fi

seed=1
skip_train=0
skip_test=0

check() {
    opt=$1
    arg=$2

    if [[ $arg =~ ^- ]] || [ ! $arg ]
    then
        echo "ERROR: -$opt expects an corresponding argument" >&2
        usage
    fi
}

while getopts :c:srt opt
do
    case $opt in
    r)
        skip_train=1
        ;;
    t)
        skip_test=1
        ;;
    c)
        check c $OPTARG
        cuda=$OPTARG
        ;;
    s)
        check s $OPTARG
        seed=$OPTARG
        ;;
    :)
        echo "ERROR: -$OPTARG requires an argument" >&2
        usage
        ;;
    \?) 
        echo "ERROR: unknown option -$OPTARG" >&2
        usage
        ;;
    esac
done

if [ ! $cuda ]
then
    echo "ERROR: -c int(cuda num) is required" >&2
    usage
fi

# job  symlinks
now=$(date +"%Y%m%d_%H%M%S")
main_name='3DGS_'
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../../..

abspath=$(cd "$(dirname "$0")";pwd)
expdir=${abspath##*experiments/}

checkpoint=$ROOT"/checkpoint/"

last_name=${abspath##*/}
last_least=${abspath%/*}
penultimate_name=${last_least##*/}
job=$main_name$penultimate_name"_"$last_name

mkdir -p start_log
mkdir -p $checkpoint$expdir

if [ ! -e $abspath"/"$last_name ]
then
  echo "checkpoint to experiment soft link created."
  ln -s $checkpoint$expdir $abspath
else
    echo "checkpoint to experiment soft link already exists."
fi
 
cd $abspath"/"$last_name

if [ ! -e $checkpoint$expdir"/"$last_name ]
then
  echo "experiment to checkpoint soft link created."
  ln -s $abspath $checkpoint$expdir
else
    echo "experiment to checkpoint soft link already exists."
fi

cd $abspath
# exit 1

export CUDA_VISIBLE_DEVICES=$cuda

echo "loading the config file "$abspath"/config.yaml"

cmd="python $ROOT/render.py --config=$abspath"/config.yaml" --seed $seed"


if [ $skip_train == 1 ]
then
    echo "skip_train"
    cmd+=" --skip_train"
fi
if [ $skip_test == 1 ]
then
    echo "skip_test"
    cmd+=" --skip_test"
fi

cmd+=" 2>&1 | tee start_log/3dgs_$now.txt"

echo "Running command: $cmd"  
eval $cmd