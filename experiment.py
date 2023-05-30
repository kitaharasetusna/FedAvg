import os



'''
save time for wrting a bunch of commands
'''
# define the command to run script1.py with options
command = 'python ./main.py -E 2 --algo fedavg' 

# run the command using subprocess
os.system(command)


# define the command to run script1.py with options
command = 'python ./main.py -E 2 --algo fedopt' 

# run the command using subprocess
os.system(command)