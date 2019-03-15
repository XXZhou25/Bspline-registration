# Registration
Bspline registration

This code can be implemented in command line.

1. Please download the zip file on your computer and unzip it.

2. In the terminal, use 'cd' to switch to your current location of the folder.

3. There are two parameters should be input into the code, 
    '-f': the fixed image's location
    '-m': the moving image's location
    
4. The whole command you input is like this:

python bspline_imagefinal2.py -f /Users/XXXXXX/Registration-master/inputs/fixed1.png -m /Users/XXXXXX/Registration-master/inputs/moving1.png

5. Then you could see the output loss in the terminal. When the iteration ends, you will see the registered result in the location. 

6. Please feel free to change the 'gridspace', 'iteration' - optimization iterations, 'lr'-learning rate, 'lamda'- coefficient of regularization term if needed.
