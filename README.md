# The Impact of Smartphone Distraction on Mother-Infant Neural Synchrony During Social Interactions :woman_feeding_baby::brain:

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
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
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

To run the validation_script on multiple cores use the following command:

```bash
mpirun -np <num_processes> python validation_script.py
```

where num_processes is the number of cores you want to use. 