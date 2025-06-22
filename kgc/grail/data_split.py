import csv
import random
        

def load_data(fp, num_secrets):
    random_seed = 26483
    weight_val = 100
    data_in = []
    
    secret_verts = {}
    secrets = []
    secret_req_chance = 0.5

    initialized = False
    

    with open(fp, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
           data_in.append(row)
           if row[0] not in secret_verts:
               secret_verts[row[0]] = 0
           if row[2] not in secret_verts:
               secret_verts[row[2]] = 0
        print(row)
        random.seed(random_seed)
        random.shuffle(data_in)
        print(int(len(data_in)))

        
        
        weights = [1 / (2*weight_val)] * len(data_in)
        
        for _ in range(num_secrets):
            if random.random() < secret_req_chance and initialized:
                secret = random.choices(data_in, weights=weights, k=1)[0]
            else:
                secret = random.choices(data_in, weights=[1] * len(data_in), k=1)[0]

            initialized = True

            secrets.append(secret)
            index = data_in.index(secret)

            secret_verts[secret[0]] += 1
            secret_verts[secret[2]] += 1
            
            del data_in[index]
            del weights[index]

            for i in range(len(data_in)):
                weights[i] = secret_verts[data_in[i][0]] + secret_verts[data_in[i][2]]
            

            

            #print(secret)
            

        print(len(secrets))
        
        data_secrets = secrets #data_in[:num_secrets]
        data_private = data_in[:int(len(data_in)/3)] #data_in[num_secrets:int(len(data_in)/3)]
        data_public = data_in[int(len(data_in)/3):] #data_in[int(len(data_in)/3):]
        
        print(len(data_private))
        print(len(data_public))
    return data_public, data_private, data_secrets

def csv_write(f_path, data):
    with open(f_path, "w") as fp: 
        wr = csv.writer(fp, delimiter="\t")    
        for row in data:
            wr.writerow(row)

def write_data(dp, data_public, data_private, data_secrets):

    csv_write(dp+"train.txt", data_public)
    csv_write(dp+"new.txt", data_private)
    csv_write(dp+"secrets.txt", data_secrets)

if __name__ == "__main__":
    dp = "./data/exp_v1_33_select/"
    data_fp = dp+"train_og.txt"
    num_secrets = 150
    (data_public, data_private, data_secrets) = load_data(data_fp, num_secrets)
    write_data(dp, data_public, data_private, data_secrets)
    
