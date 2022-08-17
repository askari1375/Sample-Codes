# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:06:08 2021

@author: Amirhossein
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
from tqdm import tqdm


def train(model,
          train_loader,
          val_loader,
          optimizer,
          num_epochs,
          device,
          save_folder,
          model_name):
    
    train_log = []
    val_log = []
    
    fig_loss = plt.figure(figsize = (16, 8))
    
    ax_loss = fig_loss.add_subplot(111)
    
    train_log.append(evaluate_model(model, train_loader, device))
    val_log.append(evaluate_model(model, val_loader, device))
    
    ax_loss.set_xlim(0, num_epochs)
    ax_loss.set_ylim(0, 1.2 * max(train_log[0], val_log[0]))
    
    train_line, = ax_loss.plot([0], train_log, color = 'C0', label = 'Train')
    val_line, = ax_loss.plot([0], val_log, color = 'C1', label = 'Validation')
    
    plt.legend()        
    plt.show()
    
    
    
    vis_num_images = 16
    vis_images_per_row = 4
    np.random.seed(0)
    fig_progress = plt.figure(figsize=(16, 8))
    plt.close(fig_progress)
    gs = fig_progress.add_gridspec(int(np.ceil(vis_num_images / vis_images_per_row)), vis_images_per_row)    
    axs_list = []    
    num_rows = vis_num_images // vis_images_per_row
    for i in range(num_rows):
        for j in range(vis_images_per_row):
            ax = fig_progress.add_subplot(gs[i, j])
            ax.axis('off')
            axs_list.append(ax)
    
    create_and_save_progress(model, device, 0, save_folder, fig_progress, axs_list, train_loader, val_loader)

    best_model_save_path = os.path.join(save_folder, model_name)
    current_time = time.time()
    for epoch in range(1, num_epochs + 1):
        print("Epoch {} / {} ...".format(epoch, num_epochs))
        train_loss = []
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.get_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_log.append(np.mean(train_loss))
        
        val_loss = evaluate_model(model, val_loader, device)
        val_log.append(val_loss.item())
        
        
        if val_log[-1] == np.min(val_log):
            torch.save(model.state_dict(), best_model_save_path)
        train_line.set_xdata(range(epoch + 1))
        train_line.set_ydata(train_log)
        val_line.set_xdata(range(epoch + 1))
        val_line.set_ydata(val_log)
        
        ax_loss.set_title("Epoch {} / {}".format(epoch, num_epochs))
        fig_loss.canvas.draw()
        fig_loss.canvas.flush_events()
        
        create_and_save_progress(model, device, epoch, save_folder, fig_progress, axs_list, train_loader, val_loader)
        duration = time.time() - current_time
        print("Epoch {} / {} : Duration = {:.2f}\tTraining Loss = {:.5f}\tValidation Loss = {:.5f}".format(epoch, num_epochs, duration, train_log[-1], val_log[-1]))
        current_time = time.time()
        
    fig_loss.savefig(best_model_save_path[:-2] + "png")
    
    
def evaluate_model(model, data_loader, device):
    all_loss = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.get_loss(outputs, targets)
            all_loss.append(loss.item())
    return np.mean(all_loss)
            
            
def create_and_save_progress(model,
                             device,
                             epoch,
                             save_folder,
                             fig_progress,
                             axs_list,
                             train_loader,
                             val_loader,
                             vis_num_images = 16, vis_images_per_row = 4):

    np.random.seed(0)
    num_rows = vis_num_images // vis_images_per_row
    sample_size = vis_num_images // 2
    
    vis_image_indexes_train = np.random.choice(len(train_loader.dataset), size=sample_size, replace=False)
    vis_image_indexes_val = np.random.choice(len(val_loader.dataset), size=sample_size, replace=False)
    
    train_images = []
    val_images = []
    
    for idx in vis_image_indexes_train:
        image, target = train_loader.dataset[idx]        
        predict = get_predict(model, device, image)
        
        image = np.moveaxis(image.numpy(), 0, -1)
        target = np.moveaxis(target.numpy(), 0, -1)
        
        
        new_image = np.concatenate((image, target, predict), axis = 1)
        train_images.append(new_image)
    for idx in vis_image_indexes_val:
        image, target = val_loader.dataset[idx]
        predict = get_predict(model, device, image)
        
        image = np.moveaxis(image.numpy(), 0, -1)
        target = np.moveaxis(target.numpy(), 0, -1)
        
        new_image = np.concatenate((image, target, predict), axis = 1)
        val_images.append(new_image)
    
    for i in range(num_rows):
        for j in range(vis_images_per_row):
            ax = axs_list[i * vis_images_per_row + j]
            if j < vis_images_per_row // 2:
                ax.imshow(train_images[i * vis_images_per_row // 2 + j], cmap = 'gray')
            else:
                ax.imshow(val_images[(i - 1) * vis_images_per_row // 2 + j], cmap = 'gray')
    fig_progress.savefig(save_folder + "/progress/epoch_{}.png".format(epoch))
            
            
def get_predict(model, device, image):
    tensor_img =torch.unsqueeze(image, 0).to(device)
    predict = torch.squeeze(model(tensor_img), 0)
    predict = np.moveaxis(predict.to(torch.device('cpu')).detach().numpy(), 0, -1)
    predict[predict < 0] = 0
    predict[predict > 1] = 1
    return predict
        

def test(model, test_loader, save_path, device):
    model.eval()
    for idx in tqdm(range(len(test_loader.dataset))):
        image, target = test_loader.dataset[idx]
        predict = get_predict(model, device, image)
        image = np.moveaxis(image.numpy(), 0, -1)
        target = np.moveaxis(target.numpy(), 0, -1)
        
        new_image = np.concatenate((image, target, predict), axis = 1)
        image_name = "{}.png".format(idx + 1)
        cv2.imwrite(os.path.join(save_path, image_name), new_image * 255)