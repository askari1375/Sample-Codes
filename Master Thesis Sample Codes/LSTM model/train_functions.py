# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 18:08:17 2021

@author: Amirhossein
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
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
    train_log.append(evaluate_model(model, train_loader, device, False))
    val_log.append(evaluate_model(model, val_loader, device, False))
    ax_loss.set_xlim(0, num_epochs)
    ax_loss.set_ylim(0, 1.2 * train_log[0])
    train_line, = ax_loss.plot([0], train_log, color = 'C0', label = 'train')
    val_line, = ax_loss.plot([0], val_log, color = 'C1', label = 'validation')
    plt.legend()
    plt.show()
    
    current_time = time.time()
    best_model_save_path = os.path.join(save_folder, model_name)
    for epoch in range(1, num_epochs + 1):
        print("Epoch {} / {} ...".format(epoch, num_epochs))
        train_loss = []
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.get_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_log.append(np.mean(train_loss))
        
        val_loss = evaluate_model(model, val_loader, device)
        val_log.append(val_loss)
        if val_log[-1] == np.min(val_log):
            torch.save(model.state_dict(), best_model_save_path)
        train_line.set_xdata(range(epoch + 1))
        train_line.set_ydata(train_log)
        val_line.set_xdata(range(epoch + 1))
        val_line.set_ydata(val_log)
        ax_loss.set_title("Epoch {} / {}".format(epoch, num_epochs))
        fig_loss.canvas.draw()
        fig_loss.canvas.flush_events()
        
        duration = time.time() - current_time
        print("Epoch {} / {} : Duration = {:.2f}\tTraining Loss = {:.5f}\tValidation Loss = {:.5f}".format(epoch, num_epochs, duration, train_log[-1], val_log[-1]))
        current_time = time.time()
    
    fig_loss.savefig(best_model_save_path[:-2] + "png")




def evaluate_model(model, data_loader, device, tqdm_disable = True):
    all_loss = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(data_loader, disable = tqdm_disable):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = model.get_loss(y_pred, y)
            all_loss.append(loss.item())
    
    return np.mean(all_loss)