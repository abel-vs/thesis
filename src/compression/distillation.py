import torch
from tqdm import tqdm
import mnist
import torch.nn.functional as F
import torch.optim as optim
import compression.pruning as prune
import general
from dataset_models import DataSet

LOGGING_STEPS = 1000

# Method that trains the student model using distillation
def distillation_train_loop(
    teacher,
    student,
    train_data,
    test_data,
    distil_criterion,
    eval_criterion,
    eval_metric,
    optimizer,
    epochs,
):
    for epoch in range(epochs):

        for (data, hard_target) in tqdm(train_data, desc="Distillation Training"):
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
            test_score = 0

            for data, hard_target in tqdm(test_data, desc="Distillation Validation"):
                output = student(data)

                loss = eval_criterion(output, hard_target)
                test_loss += loss.item()

                score = eval_metric(output, hard_target)
                test_score += score

        test_loss /= len(test_data)
        test_score /= len(test_data)

        print("Epoch: {}".format(epoch))
        print("Distillation loss: {}".format(distill_loss.item()))
        print("Test loss: {}, Test score: {}".format(test_loss, test_score))

    return student


# Method that creates a student model based on the teacher model
# TODO This should be done intelligently, returning a model that is similar to the teacher model but smaller.
# For now, we just return a model with the same architecture as the teacher model
def create_student_model(teacher_model, dataset: DataSet):
    prune.magnitude_pruning_structured(teacher_model, dataset, fineTune=False)
    return teacher_model


# Method that performs the whole distillation procedure
def perform_distillation(model, dataset: DataSet,  settings: dict = None):

    # Create student model
    student_model = create_student_model(model, dataset)

    eval_criterion = dataset.criterion
    eval_metric = dataset.metric

    # TODO: Find intelligent way to set the following properties
    distil_criterion = F.mse_loss
    epochs = 1
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)


    compressed_model = distillation_train_loop(
        model,
        student_model,
        dataset.train_loader,
        dataset.test_loader,
        distil_criterion,
        eval_criterion,
        eval_metric,
        optimizer,
        epochs,
    )
    return compressed_model
