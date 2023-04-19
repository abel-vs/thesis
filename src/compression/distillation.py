import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import src.compression.pruning as prune
import general
import plot
from dataset_models import DataSet

LOGGING_STEPS = 1000


def validate(student, test_data, eval_criterion, eval_metric):
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

    return test_loss, test_score


def train(teacher, student, train_data, distil_criterion, optimizer):
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

    return distill_loss


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
    epochs=1,
    threshold=None,
):
    # If a threshold is specified, train the student model until the threshold is reached or the score decreases
    if threshold is not None:
        previous_score = 0
        while True:
            # Validate the student model
            test_loss, test_score = validate(
                student, test_data, eval_criterion, eval_metric)
            print("Test loss: {}, Test score: {}".format(test_loss, test_score))

            # If the score is above the threshold, stop training
            if test_score > threshold:
                print("Stopped training because threshold ({}) was reached: {}".format(
                    threshold, test_score))
                break

            # If the score is decreasing, stop training
            if test_score < previous_score:
                print("Stopped training because score started decreasing: from {} to {}".format(
                    previous_score, test_score))
                break
            else:
                previous_score = test_score

            # Train the student model
            distill_loss = train(teacher, student, train_data,
                                 distil_criterion, optimizer)
            print("Distillation loss: {}".format(distill_loss.item()))

    # Otherwise, train the student model for the specified number of epochs
    else:
        for epoch in range(epochs):

            print("Epoch: {}".format(epoch))

            # Train the student model
            distill_loss = train(teacher, student, train_data,
                                 distil_criterion, optimizer)
            print("Distillation loss: {}".format(distill_loss.item()))

            # Validate the student model
            test_loss, test_score = validate(
                student, test_data, eval_criterion, eval_metric)
            print("Test loss: {}, Test score: {}".format(test_loss, test_score))

    return student


# Method that creates a student model based on the teacher model
# TODO This should be done intelligently, returning a model that is similar to the teacher model but smaller.
# For now, we just return a model with the same architecture as the teacher model
def create_student_model(teacher_model, dataset: DataSet, fineTune=False):
    teacher_model = copy.deepcopy(teacher_model)
    prune.magnitude_pruning_structured(
        teacher_model, dataset, sparsity=0.5, fineTune=fineTune)
    return teacher_model


# Method that performs the whole distillation procedure
def perform_distillation(model, dataset: DataSet,  settings: dict = None):

    print("Settings:", settings)
    # Extract settings
    performance_target = settings.get("performance_target", None)
    epochs = settings.get("epochs", 1)
    sparsity = settings.get("sparsity", 0.5)
    fineTune = settings.get("fineTune", False)

    # Create student model
    plot.print_header("Creating student model")
    print("Fine-tuning:", fineTune)
    student_model = create_student_model(model, dataset, fineTune)
    plot.print_header()

    eval_criterion = dataset.criterion
    eval_metric = dataset.metric

    # TODO: Find intelligent way to set the following properties
    distil_criterion = F.mse_loss
    optimizer = optim.Adam(student_model.parameters(), lr=0.01)

    print("\n")
    plot.print_header("Performing distillation")
    compressed_model = distillation_train_loop(
        model,
        student_model,
        dataset.train_loader,
        dataset.test_loader,
        distil_criterion,
        eval_criterion,
        eval_metric,
        optimizer,
        epochs=epochs,
        threshold=performance_target,
    )
    plot.print_header()

    return compressed_model
