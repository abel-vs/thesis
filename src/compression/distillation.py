import torch

# Method that trains the student model using distillation
def distillation_train_loop(teacher, student, train_data, test_data, distil_criterion, test_criterion, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        for batch_id, (data, hard_target) in enumerate(train_data):
            # Compute the output logits of the teacher and student models
            soft_target = teacher(data)
            output = student(data)
            
            # Compute the loss and gradient
            distill_loss = distil_criterion(output, soft_target.detach())
            distill_loss.backward()
            
            # Update the student model's parameters
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            optimizer.zero_grad()

            if batch_id % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_id * len(data), len(train_data.dataset),
                    100. * batch_id / len(train_data), loss.item()))
                
        # Validate the student model
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for data, hard_target in test_data:
                logits = student(data)

                loss = test_criterion(logits, hard_target)
                test_loss += loss.item()

                _, predicted = logits.max(1)
                correct += (predicted == hard_target).sum().item()
                total += hard_target.size(0)

        test_acc = correct / total
        test_loss /= len(test_data)
        print('Epoch: {}, Test loss: {}, Test accuracy: {}'.format(epoch, test_loss, test_acc))


# Method that creates a student model based on the teacher model
# TODO This should be done intelligently, returning a model that is similar to the teacher model but smaller.
# For now, we just return a model with the same architecture as the teacher model
def create_student_model(teacher_model):
    return teacher_model

