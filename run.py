import subprocess

with open('train_option.txt', 'r') as file:
    commands = file.readlines()

for command in commands:
    process = subprocess.Popen(command, shell=True)
    process.communicate()