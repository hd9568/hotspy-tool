import myFunction
import sys
import os

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    parts = content.split("\n\n")  
    if len(parts) >= 3:
        return parts[2] 
    else:
        return "文件不包含足够的部分。"


def getAttributes(string):
    elements = []
    current_element = ""
    for char in string:
        if char.isspace():
            if(current_element != ""):
               elements.append(current_element)
            current_element = ""
        else:
            current_element += char

    if(current_element != ""):
        elements.append(current_element)
    return elements

fileFolder = str(sys.argv[1])+"_data"
report = str(sys.argv[2])

print(fileFolder+" "+report)

filename = report
data = read_file(filename)
#print(third_part_content)

if not os.path.exists(fileFolder):
        os.makedirs(fileFolder)
full_file_path = os.path.join(fileFolder, fileFolder+".txt")
file = open(full_file_path, 'w', encoding='utf-8')
#with open(full_file_path, 'w', encoding='utf-8') as file:


lines = data.split('\n')
functions = []  
i = 0
for line in lines:
    parts = getAttributes(line)
    if len(parts) >= 7:
        # print(parts)
        i = i + 1
        func = myFunction.Function(type_=parts[0], max_buf=parts[1], visits=parts[2], 
                            time=parts[3], time_percent=parts[4], time_per_visit=parts[5], 
                            region=' '.join(parts[6:]))
        if(func.get_type()=="USR"):
            file.write(str(func.get_region())+" "+str(func.get_hotpot())+"\n") 
if(len(lines)!= i):
     print("usr number may be wrong!")
