#!/bin/bash
echo "[begin generating dataset]"

echo "Convert medical images to numpy.."
python data/to_numpy.py
echo "Done!"

echo "Partition 3D images into patches.."
python data/partition_data.py
echo "Done!"

echo "Create data list.."
python data/create_list.py --number $1
echo "Done!"

echo "Calculate dataset statistics.."
python data/statistic.py
echo "Done!"

echo "[dataset is successfully generated.]"

echo "Create directories.."
mkdir -p checkpoints
mkdir -p LPCNN/result
mkdir -p LPCNN/tb_log
mkdir -p LPCNN/test_result
