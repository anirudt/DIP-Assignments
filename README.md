## What?
This repo contains my submissions to the assignments in the course *Digital Image Processing*.

## How to run the second assignment?
```
cd ass2/
python -m cProfile interpol.py -b | grep interpol
```
This command will run the cProfile module on the NNinterpol method and grep for the function profiling times.
If you want to list all the options that you could run instead of -b, run a 
```
python interpol.py -h
```
Alternatively, following are the options:
```
-a: read_czp
-b: NNinterpol
-c: BLinterpol
-d: pol2cart
-e: cart2pol
```
The cart2pol and pol2cart functions are implemented both in vectorized and non vectorized forms, this could be toggled by changing the last parameter. 
