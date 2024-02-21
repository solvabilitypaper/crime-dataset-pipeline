This package contains the crime dataset pipeline.

## Installation
This project is written in `python3` (3.10.x) and can be installed as follows:

```bash
python -m venv env_py310_crimedataset
source ./env_py310_crimedataset/bin/activate
pip install --upgrade pip setuptools wheel requests
git clone ... crime-dataset-pipeline
cd crime-dataset-pipeline
pip install -r requirments.txt
```

The package requires MongoDB Community Edition. Please install it through the 
package manager of user operating systems:
https://www.mongodb.com/docs/manual/administration/install-on-linux/

## Datasets
We do not provide any crime datasets in this repository. The following files
need to be downloaded manually:

Crime_Data_from_2010_to_2019.csv: 
https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z

Crime_Data_from_2020_to_Present.csv:
https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8

These files need to be stored in the following folder: `./data/los_angeles/`

Please note that the software package will download further data from
OpenStreetMap.


## Usage

For executing the pipeline and to generate the dataset, the following command
needs to be executed:
```bash
python main.py
```
The output is directly stored in the MongoDB.

For generating the knowledge graph from the MongoDB, the following command 
needs to be executed:
```bash
python experiments/step1_main_build_graph.py
```

For learning the knowledge graph embeddings, please execute the following command:
```bash
python experiments/step2_learn_embeddings.py
```

The tabular experiments can be executed after the knowledge graph was generated:
```bash
python experiments/step3_experiments_tabular.py
```

The graph experiments can be executed after the knowledge graph was generated:
```bash
python experiments/step3_experiments_graph.py
```

Finally, the raw results can be parsed and aggregated with the following command:
```bash
python experiments/step4_result_parser.py
```

## Troubleshooting
### Data Folder
Please note that you might have to create or adapt some path variables.
It is assumed that the following location exists: `/data/crime-knowledge-graph/`.
If this is not the case, please create it or adapt the python files accordingly.

### Project Folder
If you experience import errors of our own python classes, please add the project
location to your system PATH variable. For example: `sys.path.append('/root/crime-knowledge-graph/src')`
