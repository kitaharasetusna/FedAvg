import os



'''
save time for wrting a bunch of commands
'''
# define the command to run script1.py with options
command = 'python ./main.py -T 40 -E 1 -B 10 --algo fedavg --num_client 100' 

# run the command using subprocess
os.system(command)


# # define the command to run script1.py with options
# command = 'python ./main.py -T 20 -E 1 -B 10 --algo fedopt --num_client 100' 

# # run the command using subprocess
# os.system(command)

# # define the command to run script1.py with options
# command = 'python ./main.py -T 20 -E 1 -B 10 --algo fedadag --num_client 100' 

# # run the command using subprocess
# os.system(command)