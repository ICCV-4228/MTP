import yaml
import io
def main():
    data = {'dataset':'H36M','node_num':25,'in_features':9, 'out_features':25,
            'hidden_size':128,'learning_rate':0.001, 'batch_size':16, 'train_epoches':10,
            'gradient_clip':5.0,'epsilon':0.01, 'input_n':10, 'output_n':25}
    with open('./config.yml', 'w') as f:
        yaml.dump(data, f,allow_unicode=True)
        print('writen complete!')
if __name__ == '__main__':
    main()