import torch

# Method that trains the student model using distillation
def distillation_train_loop(student, teacher, train_data, val_data, loss_fn, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        for x, y in train_data:
            # Compute the output logits of the teacher and student models
            logits_teacher = teacher(x)
            logits_student = student(x)
            
            # Compute the loss and gradient
            loss = loss_fn(logits_student, logits_teacher.detach())
            loss.backward()
            
            # Update the student model's parameters
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()
        
        # Validate the student model
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in val_data:
                logits = student(x)
                _, predicted = logits.max(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
            val_acc = correct / total
        print('Epoch: {}, Validation accuracy: {}'.format(epoch, val_acc))


# Method that creates a student model based on the teacher model
# TODO This should be done intelligently, returning a model that is similar to the teacher model but smaller.
# For now, we just return a model with the same architecture as the teacher model
def create_student_model(teacher_model):
    return teacher_model