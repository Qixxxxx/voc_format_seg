import os 


with open('./valSD01.txt', 'w') as f:
    after_generate = os.listdir("./SD01")
    for image in after_generate:
        if len(image.split(".")[0]) == 4:
            f.write(image.split(".")[0] +"\n")


with open('./valSD02.txt', 'w') as f:
    after_generate = os.listdir("./SD02")
    for image in after_generate:
        if len(image.split(".")[0]) == 4:
            f.write(image.split(".")[0] +"\n")


with open('./valSD03.txt', 'w') as f:
    after_generate = os.listdir("./SD03")
    for image in after_generate:
        if len(image.split(".")[0]) == 4:
            f.write(image.split(".")[0] +"\n")




