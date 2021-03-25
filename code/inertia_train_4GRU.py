from __future__ import print_function, absolute_import, division
import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import DataLoader
import data_utils
import space_angle_velocity
import bone_length_loss
import model_4GRU
import torchsnooper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)

node_num = config['node_num']
input_n=config['input_n']
output_n=config['output_n']
base_path = './data'
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = config['out_features']
lr=config['learning_rate']
batch_size = config['batch_size']


train_save_path = os.path.join(base_path, 'train.npy')
train_save_path = train_save_path.replace("\\","/")
dataset = np.load(train_save_path,allow_pickle = True)

chain = [[1], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
         [233.58, 257.08, 121.13, 115], [257.08, 151.03, 278.88, 251.73, 100 ],
         [257.08,151.03, 278.88, 251.73, 100]]
for x in chain:
    s = sum(x)
    if s == 0:
        continue
    for i in range(len(x)):
        x[i] = (i+1)*sum(x[i:])/s

chain = [item for sublist in chain for item in sublist]
nodes_weight = torch.tensor(chain)
nodes_weight = nodes_weight.unsqueeze(1)
nodes_frame_weight = nodes_weight.expand(25, 25)

frame_weight = torch.tensor([3, 2, 1.5, 1.5, 1, 0.5, 0.2, 0.2, 0.1, 0.1,
                             0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02,
                             0.02, 0.02, 0.02, 0.02])



for epoch in range(config['train_epoches']):

    for i in range (dataset.shape[0]):
        data = dataset[i]

        train_data = data_utils.LPDataset(data, input_n, output_n)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        model_x = model_4GRU.Generator(input_size, hidden_size, output_size, node_num, batch_size)
        model_y = model_4GRU.Generator(input_size, hidden_size, output_size, node_num, batch_size)
        model_z = model_4GRU.Generator(input_size, hidden_size, output_size, node_num, batch_size)
        model_v = model_4GRU.Generator(input_size, hidden_size, output_size, node_num, batch_size)

        mse = nn.MSELoss(reduction='mean')
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model_v.parameters()) / 1000000.0))

        optimizer_x = optim.Adam(model_x.parameters(), lr)
        optimizer_y = optim.Adam(model_y.parameters(), lr)
        optimizer_z = optim.Adam(model_z.parameters(), lr)
        optimizer_v = optim.Adam(model_v.parameters(), lr)

        print('pretrain generator')
        if os.path.exists(os.path.join(base_path, 'generator_x_4GRU.pkl')):
            print('---------------------------------')
            model_x.load_state_dict(torch.load(os.path.join(base_path, 'generator_x_4GRU.pkl')),strict=False)
            model_y.load_state_dict(torch.load(os.path.join(base_path, 'generator_y_4GRU.pkl')),strict=False)
            model_z.load_state_dict(torch.load(os.path.join(base_path, 'generator_z_4GRU.pkl')),strict=False)
            model_v.load_state_dict(torch.load(os.path.join(base_path, 'generator_v_4GRU.pkl')),strict=False)
            for i, data in enumerate(train_loader):
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()
                optimizer_v.zero_grad()

                in_shots, out_shot = data
                input_angle = in_shots[:, 1:, :, :3]
                input_velocity = in_shots[:, 1:, :, 3].permute(0, 2, 1)
                target_angle = out_shot[:, :, :, :3]
                target_velocity = out_shot[:, :, :, 3]

                #read velocity
                input_velocity = input_velocity.float()
                target_velocity = target_velocity.float()

                #read angle_x
                input_angle_x = input_angle[:,:,:,0].permute(0, 2, 1).float()
                target_angle_x = target_angle[:,:,:,0].float()
                
                
                

                #read angle_y
                input_angle_y = input_angle[:,:,:,1].permute(0, 2, 1).float()
                target_angle_y = target_angle[:,:,:,1].float()

                #read angle_z
                input_angle_z = input_angle[:,:,:,2].permute(0, 2, 1).float()
                target_angle_z = target_angle[:,:,:,2].float()

                #read 3D data
                input_3d_data = in_shots[:, :, :, 4:]
                target_3d_data =out_shot[:, :, :, 4:]

                loss_v = 0
                loss_x = 0
                loss_y = 0
                loss_z = 0
                loss_rec = 0

                output_v, _ = model_v(input_velocity,  hidden_size)
                
                output_v = output_v.view(target_velocity.shape[0],target_velocity.shape[2],output_size)
                target_velocity_loss = target_velocity.permute(0, 2, 1)
                loss_v += torch.mean(torch.norm((output_v- target_velocity_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_x, _ = model_x(input_angle_x, hidden_size)
                output_x = output_x.view(target_angle_x.shape[0],target_angle_x.shape[2],output_size)
                target_angle_x_loss = target_angle_x.permute(0, 2, 1)
                loss_x += torch.mean(torch.norm((output_x- target_angle_x_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_y, _ = model_y(input_angle_y, hidden_size)
                output_y = output_y.view(target_angle_y.shape[0],target_angle_y.shape[2],output_size)
                target_angle_y_loss = target_angle_y.permute(0, 2, 1)
                loss_y += torch.mean(torch.norm((output_y- target_angle_y_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_z, _ = model_z(input_angle_z, hidden_size)
                output_z = output_z.view(target_angle_z.shape[0],target_angle_z.shape[2],output_size)
                target_angle_z_loss = target_angle_z.permute(0, 2, 1)
                loss_z += torch.mean(torch.norm((output_z- target_angle_z_loss)*frame_weight*nodes_frame_weight, 2, 1))

                angle_x = output_x.permute(0, 2, 1)
                angle_y = output_y.permute(0, 2, 1)
                angle_z = output_z.permute(0, 2, 1)
                pred_v = output_v.permute(0, 2, 1)
                
                pred_angle_set = torch.stack((angle_x,angle_y,angle_z),3)

                pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0],pred_angle_set.shape[1],-1,3)

                #reconstruction_loss
                # input_pose = torch.zeros((target_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))
                # for a in range(input_pose.shape[0]):
                    # input_pose[a,0,:,:] = input_3d_data[a,input_n-1,:,:]
                # re_data = torch.FloatTensor([])
                # for b in range (target_3d_data.shape[0]):
                    # for c in range (target_3d_data.shape[1]):
                        # print('input_pose:\n',input_pose[j, i, :, :])
                        # print('pred_v:\n',pred_v[j,i,:,])
                        # reconstruction_coordinate = space_angle_velocity.reconstruction_motion(pred_v[b,c,:,], pred_angle_set[b, c,:,:], input_pose[b, c, :, :])
                        # re_data = torch.cat([re_data,reconstruction_coordinate],dim=0)
                        # reconstruction_coordinate = reconstruction_coordinate
                        # print('reconstruction_coordinate.shape:\n',reconstruction_coordinate.shape)
                        # if c+1<target_3d_data.shape[1]:
                            # input_pose[b,c+1,:,:] = reconstruction_coordinate
                        # else:
                            # continue
                        # print('input_pose[j]:\n',input_pose[j].shape)
                # re_data = re_data.view(target_3d_data.shape[0],-1,17,3)
                # bone_loss = bone_length_loss.bone_length_loss(re_data, target_3d_data)
                # loss_rec = torch.mean(torch.norm((re_data - target_3d_data)*nodes_3d_weight, 2, 1))
                total_loss = 100000*loss_v + 10000*loss_x + 10000*loss_y + 10000*loss_z
                total_loss.backward()
                nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_v.parameters(), config['gradient_clip'])

                optimizer_x.step()
                optimizer_y.step()
                optimizer_z.step()
                optimizer_v.step()
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
            torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x_4GRU.pkl'))
            torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y_4GRU.pkl'))
            torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z_4GRU.pkl'))
            torch.save(model_v.state_dict(), os.path.join(base_path, 'generator_v_4GRU.pkl'))


        else:
            for i, data in enumerate(train_loader):
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()
                optimizer_v.zero_grad()

                in_shots, out_shot = data

                input_angle = in_shots[:, 1:, :, :3]
                input_velocity = in_shots[:, 1:, :, 3].permute(0, 2, 1)
                target_angle = out_shot[:, :, :, :3]
                target_velocity = out_shot[:, :, :, 3]

                #read velocity
                input_velocity = input_velocity.float()
                target_velocity = target_velocity.float()

                #read angle_x
                input_angle_x = input_angle[:,:,:,0].permute(0, 2, 1).float()
                target_angle_x = target_angle[:,:,:,0].float()
                
                #read angle_y
                input_angle_y = input_angle[:,:,:,1].permute(0, 2, 1).float()
                target_angle_y = target_angle[:,:,:,1].float()

                #read angle_z
                input_angle_z = input_angle[:,:,:,2].permute(0, 2, 1).float()
                target_angle_z = target_angle[:,:,:,2].float()

                #read 3D data
                input_3d_data = in_shots[:, :, :, 4:]
                target_3d_data =out_shot[:, :, :, 4:]

                loss_v = 0
                loss_x = 0
                loss_y = 0
                loss_z = 0
                loss_rec = 0
               
                output_v, _ = model_v(input_velocity,  hidden_size)
                
                output_v = output_v.view(target_velocity.shape[0],target_velocity.shape[2],output_size)
                target_velocity_loss = target_velocity.permute(0, 2, 1)
                loss_v += torch.mean(torch.norm((output_v- target_velocity_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_x, _ = model_x(input_angle_x, hidden_size)
                output_x = output_x.view(target_angle_x.shape[0],target_angle_x.shape[2],output_size)
                target_angle_x_loss = target_angle_x.permute(0, 2, 1)
                loss_x += torch.mean(torch.norm((output_x- target_angle_x_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_y, _ = model_y(input_angle_y, hidden_size)
                output_y = output_y.view(target_angle_y.shape[0],target_angle_y.shape[2],output_size)
                target_angle_y_loss = target_angle_y.permute(0, 2, 1)
                loss_y += torch.mean(torch.norm((output_y- target_angle_y_loss)*frame_weight*nodes_frame_weight, 2, 1))

                output_z, _ = model_z(input_angle_z, hidden_size)
                output_z = output_z.view(target_angle_z.shape[0],target_angle_z.shape[2],output_size)
                target_angle_z_loss = target_angle_z.permute(0, 2, 1)
                loss_z += torch.mean(torch.norm((output_z- target_angle_z_loss)*frame_weight*nodes_frame_weight, 2, 1))

                angle_x = output_x.permute(0, 2, 1)
                angle_y = output_y.permute(0, 2, 1)
                angle_z = output_z.permute(0, 2, 1)
                pred_v = output_v.permute(0, 2, 1)
                
                pred_angle_set = torch.stack((angle_x,angle_y,angle_z),3)
                pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0],pred_angle_set.shape[1],-1,3)

                #reconstruction_loss
                # input_pose = torch.zeros((target_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))
                # for a in range(input_pose.shape[0]):
                    # input_pose[a,0,:,:] = input_3d_data[a,input_n-1,:,:]
                # re_data = torch.FloatTensor([])
                # for b in range (target_3d_data.shape[0]):
                    # for c in range (target_3d_data.shape[1]):
                        # print('input_pose:\n',input_pose[j, i, :, :])
                        # print('pred_v:\n',pred_v[j,i,:,])
                        # reconstruction_coordinate = space_angle_velocity.reconstruction_motion(pred_v[b,c,:,], pred_angle_set[b, c,:,:], input_pose[b, c, :, :])
                        # re_data = torch.cat([re_data,reconstruction_coordinate],dim=0)
                        # reconstruction_coordinate = reconstruction_coordinate
                        # print('reconstruction_coordinate.shape:\n',reconstruction_coordinate.shape)
                        # if c+1<target_3d_data.shape[1]:
                            # input_pose[b,c+1,:,:] = reconstruction_coordinate
                        # else:
                            # continue
                        # print('input_pose[j]:\n',input_pose[j].shape)
                # re_data = re_data.view(target_3d_data.shape[0],-1,17,3)
                # bone_loss = bone_length_loss.bone_length_loss(re_data, target_3d_data)
                # loss_rec = torch.mean(torch.norm((re_data - target_3d_data)*nodes_3d_weight, 2, 1))
                total_loss = 100000*loss_v + 10000*loss_x + 10000*loss_y + 10000*loss_z

                total_loss.backward()
                nn.utils.clip_grad_norm_(model_x.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_y.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_z.parameters(), config['gradient_clip'])
                nn.utils.clip_grad_norm_(model_v.parameters(), config['gradient_clip'])

                optimizer_x.step()
                optimizer_y.step()
                optimizer_z.step()
                optimizer_v.step()
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, total_loss.item()))
            torch.save(model_x.state_dict(), os.path.join(base_path, 'generator_x_4GRU.pkl'))
            torch.save(model_y.state_dict(), os.path.join(base_path, 'generator_y_4GRU.pkl'))
            torch.save(model_z.state_dict(), os.path.join(base_path, 'generator_z_4GRU.pkl'))
            torch.save(model_v.state_dict(), os.path.join(base_path, 'generator_v_4GRU.pkl'))

torch.save(model_x, os.path.join(base_path, 'generator_x_4GRU.pkl'))
torch.save(model_y, os.path.join(base_path, 'generator_y_4GRU.pkl'))
torch.save(model_z, os.path.join(base_path, 'generator_z_4GRU.pkl'))
torch.save(model_v, os.path.join(base_path, 'generator_v_4GRU.pkl'))
print ('Parameters are stored in the generator.pkl file')


















































