import torch
from tqdm import tqdm
import mnist
import torch.nn.functional as F
import torch.optim as optim

LOGGING_STEPS = 1000

# Method that trains the student model using distillation


def distillation_train_loop(
    teacher,
    student,
    train_data,
    test_data,
    distil_criterion,
    test_criterion,
    optimizer,
    epochs,
):
    for epoch in range(epochs):

        for batch_id, (data, hard_target) in enumerate(
            tqdm(train_data, desc="Distillation Training")
        ):
            # Compute the output logits of the teacher and student models
            soft_target = teacher(data)
            output = student(data)

            optimizer.zero_grad()

            # Compute the loss and gradient
            distill_loss = distil_criterion(output, soft_target.detach())
            distill_loss.backward()

            # Update the student model's parameters
            optimizer.step()

        # Validate the student model
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for data, hard_target in tqdm(test_data, desc="Distillation Validation"):
                output = student(data)

                loss = test_criterion(output, hard_target)
                test_loss += loss.item()

                _, predicted = output.max(1)
                correct += (predicted == hard_target).sum().item()
                total += hard_target.size(0)

        test_acc = correct / total
        test_loss /= len(test_data)

        print("Epoch: {}".format(epoch))
        print("Distillation loss: {}".format(distill_loss.item()))
        print("Test loss: {}, Test accuracy: {}".format(test_loss, test_acc))

    return student


# Method that creates a student model based on the teacher model
# TODO This should be done intelligently, returning a model that is similar to the teacher model but smaller.
# For now, we just return a model with the same architecture as the teacher model
def create_student_model(teacher_model):
    return teacher_model


def example_distil_loop(model):

    student_model = mnist.MnistSmallLinear()

    epochs = 5
    lr = 0.01

    optimizer = optim.Adam(
        student_model.parameters(), lr=lr
    )  # Important: use the student model parameters
    distil_criterion = F.mse_loss
    eval_criterion = F.cross_entropy

    compressed_model = distillation_train_loop(
        model,
        student_model,
        mnist.train_loader,
        mnist.test_loader,
        distil_criterion,
        eval_criterion,
        optimizer,
        epochs,
    )
    return compressed_model
