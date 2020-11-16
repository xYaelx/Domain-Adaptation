import torch
import time

class Trainer:
    def __init__(self, device, domain1_dataloader, domain2_dataloader, batch_size):
        self.device = device
        self.domain1_dataloader = domain1_dataloader
        self.domain2_dataloader = domain2_dataloader
        self.batch_size = batch_size

    def binary_acc(self,y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        #     acc = torch.round(acc * 100)

        return acc

    def train_model(self, use_discriminator, training_params, writer=None):
        since = time.time()

        print("Starting epochs")
        for epoch in range(1, training_params.num_epochs + 1):
            print(f'Epoch: {epoch} of {training_params.num_epochs}')
            training_params.model.train()  # Set model to training mode
            running_corrects = 0.0
            running_corrects_domain = 0.0

            join_dataloader = zip(self.domain1_dataloader.data['train'],
                                  self.domain2_dataloader.data['train'])  # TODO check how females_data is built
            for i, ((domain1_x, domain1_label), (domain2_x, _)) in enumerate(join_dataloader):
                # data['train'] contains (domain1_x, domain1_y) for every batch (so i=[1...NUM OF BATCHES])
                samples = torch.cat([domain1_x, domain2_x])
                samples = samples.to(self.device)
                label_y = domain1_label.to(self.device)
                domain_y = torch.cat([torch.ones(domain1_x.shape[0]), torch.zeros(domain2_x.shape[0])])
                domain_y = domain_y.to(self.device)

                # zero the parameter gradients
                training_params.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    label_preds = training_params.model(samples)[
                                  :domain1_x.shape[0]]  # TODO check if x[:domain1_x.shape[0]] = domain1_x
                    label_loss = training_params.label_criterion(label_preds, label_y)

                    extracted_features = training_params.model.avgpool.activation[
                        'avgpool']  # Size: torch.Size([16, 512, 1, 1])
                    extracted_features = extracted_features.view(extracted_features.shape[0], -1)
                    domain_preds = training_params.model.discriminator(extracted_features).squeeze()

                    if use_discriminator:
                        domain_loss = training_params.domain_criterion(domain_preds, domain_y)
                    else:
                        domain_loss = 0

                    loss = label_loss + domain_loss
                    # backward + optimize only if in training phase
                    loss.backward()
                    training_params.optimizer.step()

                batch_loss = loss.item() * samples.size(0)
                running_corrects += torch.sum(label_preds.max(1)[1] == label_y.data).item()
                running_corrects_domain += self.binary_acc(domain_preds, domain_y.data)

                if writer is not None:  # save train label_loss for each batch
                    x_axis = 1000 * (epoch + i / (self.domain1_dataloader.dataset_size[
                                                      'train'] // self.batch_size))  # TODO devidie by batch size or batch size//2?
                    writer.add_scalar('batch label_loss', batch_loss / self.batch_size, x_axis)

            if training_params.scheduler is not None:
                training_params.scheduler.step()  # scheduler step is performed per-epoch in the training phase

            train_acc = running_corrects / self.domain1_dataloader.dataset_size[
                'train']  # TODO change the accuracy ratio by the relevant dataset
            train_acc_domain = running_corrects_domain / i  # avg accuracy per epoch (i= num. of batches)

            epoch_loss, epoch_acc = self.eval_model(self.domain1_dataloader, training_params)

            if writer is not None:  # save epoch accuracy
                x_axis = epoch
                writer.add_scalar('accuracy-train', train_acc, x_axis)
                writer.add_scalar('accuracy-val', epoch_acc, x_axis)
                writer.add_scalar('accuracy-train-discriminator', train_acc_domain, x_axis)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # return the last trained model
        return training_params

    def eval_model(self, dataloader, training_params):
        training_params.model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0.0

        for i, (inputs, labels) in enumerate(dataloader.data['val']):
            # data['val'] contains (input,labels) for every batch (so i=[1...NUM OF BATCHES]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            training_params.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = training_params.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = training_params.label_criterion(outputs, labels)

            # statistics - sum loss and accuracy on all batches
            running_loss += loss.item() * inputs.size(0)  # item.loss() is the average loss of the batch
            running_corrects += torch.sum(outputs.max(1)[1] == labels.data).item()

        epoch_loss = running_loss / dataloader.dataset_size['val']
        epoch_acc = running_corrects / dataloader.dataset_size['val']
        print(f'Test Loss: {epoch_loss:.4f} TestAcc: {epoch_acc:.4f}')
        return epoch_loss, epoch_acc

    def test(self, dataloader, training_params):
        test_loss, test_acc = self.eval_model(dataloader, training_params)
        print(f'Test accuracy: {test_acc:.4f}')
        return test_acc
