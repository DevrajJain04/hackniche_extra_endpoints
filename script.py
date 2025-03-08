#read file 
with open("your_script.txt") as f:
    script_content = f.read() 
# remove newline character
script_content = script_content.replace("\n", " ")
# store in a file
with open("script.txt", "w") as f:
    f.write(script_content)

