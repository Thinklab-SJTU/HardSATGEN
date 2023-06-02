import os


source_path = './dataset/pigeonhole'
store_csv = './dataset/pi_stats.csv'


for filename in os.listdir(source_path):
    if not filename[-4:] == ".cnf":
        continue
    lcg_filename = filename.split(".")[0]
    with open(f'{source_path}/{filename}') as cnf:
        content = cnf.readlines()
        while content[0].split()[0] == 'c':
            content = content[1:]
        if content[0].split()[0] == 'p':
            num_vars = int(content[0].split(' ')[2])
            num_clauses = int(content[0].split(' ')[3])
    
    with open(store_csv, 'a') as out_file:
        out_file.write("pigeonhole/{},{},{}\n".format(lcg_filename, num_vars, num_clauses))
