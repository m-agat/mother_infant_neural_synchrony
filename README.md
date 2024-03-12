# The Impact of Smartphone Distraction on Mother-Infant Neural Synchrony During Social Interactions :woman_feeding_baby::brain:

### File descriptions

1. load_data.py ---> loading participants data
2. connectivity_measures.py ---> classes computing plv, pli, and wpli
3. validation_script.py ---> script validating the synchrony between the mother and infant
4. topographical_analysis.py ---> analysis of the differences in synchrony between the selected brain regions
5. statistical_analysis.py ---> functions for performing a statistical analysis of the results 
6. data_analysis.ipynb ---> a notebook with results and visualizations of the analysis

### Running the code for the first time

1. Create virtual env
```bash
python3 -m venv venv/
```
2. Open virtual env
```bash
source venv/bin/activate
```
3. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running the next
command, it should print a list with installed dependencies
```bash
pip list
```
4. Close virtual env
```bash
deactivate
```

### Executing the scripts

1. Open virtual env
```bash
source venv/bin/activate
```

1. Run the selected file
```bash
python <filename>.py
```

To run the validation_script on multiple cores use the following command:

```bash
mpirun -np <num_processes> python validation_script.py
```

where num_processes is the number of cores you want to use. 
