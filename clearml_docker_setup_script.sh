mkdir -p /datasets/iccv
clearml-data get --id 04fe1e704c1a424bbe941d8f95f23fd8
ln -s /clearml_agent_cache/storage_manager/datasets/ds_04fe1e704c1a424bbe941d8f95f23fd8 /datasets/iccv/val
clearml-data get --id ac705e35976947d3ace2283fbb11a87b
ln -s  /clearml_agent_cache/storage_manager/datasets/ds_ac705e35976947d3ace2283fbb11a87b /datasets/iccv/val
clearml-data get --id 2f8e31b466494869bddc88d100eded5c
ln -s /clearml_agent_cache/storage_manager/datasets/ds_2f8e31b466494869bddc88d100eded5c /datasets/iccv/train
pip install tlt