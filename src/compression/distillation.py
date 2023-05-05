import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import src.compression.pruning as prune
import general
import plot
from dataset_models import DataSet


def soft_target_distillation(teacher, student, dataset, distil_criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    teacher.eval()  # Set teacher model to evaluation mode
    student.train()  # Set student model to training mode

    for data, _ in tqdm(dataset.train_loader, desc="Distillation Training"):
        # Move data to the appropriate device
        data = data.to(device)

        # Compute the soft target probabilities from the teacher model
        with torch.no_grad():
            teacher_output = teacher(data)

        # Compute the output probabilities of the student model
        student_output = student(data)

        # Zero the gradients before computing the loss
        optimizer.zero_grad()

        # Compute the loss and gradient
        distill_loss = distil_criterion(student_output, teacher_output.detach())
        distill_loss.backward()

        # Update the student model's parameters
        optimizer.step()

    return distill_loss


def combined_loss_distillation(teacher, student, dataset, distil_criterion, optimizer, alpha=0, temperature=1.5):
    device = general.get_device()
    
    teacher.eval()  # Set teacher model to evaluation mode
    student.train()  # Set student model to training mode

    task_criterion = dataset.criterion

    for data, hard_target in tqdm(dataset.train_loader, desc="Distillation Training"):
        # Move data and hard_target to the appropriate device
        data = data.to(device)
        hard_target = hard_target.to(device)

        # Compute the soft target probabilities from the teacher model
        with torch.no_grad():
            teacher_output = teacher(data)
            teacher_soft_target = F.softmax(teacher_output / temperature, dim=1)

        # Compute the output probabilities of the student model
        student_output = student(data)
        student_soft_output = F.softmax(student_output / temperature, dim=1)

        # Zero the gradients before computing the loss
        optimizer.zero_grad()

        # Compute the task loss (e.g., cross-entropy loss for classification tasks)
        task_loss = task_criterion(student_output, hard_target)

        # Compute the distillation loss (e.g., KL Divergence for soft-target distillation)
        distill_loss = distil_criterion(student_soft_output, teacher_soft_target.detach())

        # Combine the task loss and distillation loss using the weight alpha
        combined_loss = (1 - alpha) * task_loss + alpha * distill_loss
        combined_loss.backward()

        # Update the student model's parameters
        optimizer.step()

    return combined_loss


# Method that trains the student model using distillation
def distillation_train_loop(
    teacher,
    student,
    dataset: DataSet,
    distil_technique,
    distil_criterion,
    optimizer,
    epochs=1,
    threshold=None,
):
    device = general.get_device()
    teacher.to(device)
    student.to(device)

    # If a threshold is specified, train the student model until the threshold is reached or the score decreases
    if threshold is not None:
        previous_score = 0
        while True:
            # Validate the student model
            validation_metrics = general.validate(student, dataset)
            validation_score = validation_metrics[0]

            # If the score is above the threshold, stop training
            if validation_score > threshold:
                print("Stopped training because threshold ({}) was reached: {}".format(
                    threshold, validation_score))
                break

            # If the score is decreasing, stop training
            if validation_score < previous_score:
                print("Stopped training because score started decreasing: from {} to {}".format(
                    previous_score, validation_score))
                break
            else:
                previous_score = validation_score

            # Train the student model
            distil_technique(teacher, student, dataset, distil_criterion, optimizer)

    # Otherwise, train the student model for the specified number of epochs
    else:
        for e in range(epochs):

            # Train the student model
            distil_technique(teacher, student, dataset, distil_criterion, optimizer)

            # Validate the student model
            general.validate(student, dataset)

    return student


# Method that creates a student model based on the teacher model
def create_student_model(teacher_model, dataset: DataSet, fineTune=True):
    teacher_model = copy.deepcopy(teacher_model)
    prune.channel_pruning(
        teacher_model, dataset, sparsity=0.5, fineTune=fineTune)
    return teacher_model


# Method that performs the whole distillation procedure
def perform_distillation(model, dataset: DataSet, student_model=None,  settings: dict = None):

    # Extract settings
    performance_target = settings.get("performance_target", None)
    epochs = settings.get("epochs", 1)
    distil_technique = settings.get("distil_technique", combined_loss_distillation)
    distil_criterion = settings.get("distil_criterion", F.cross_entropy)

    # Create student model if not provided
    if student_model is None:
        plot.print_header("Creating student model")
        student_model = create_student_model(model, dataset)
        plot.print_header()

    # TODO: Find intelligent way to set the following properties
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    print("\n")
    plot.print_header("Performing distillation")
    compressed_model = distillation_train_loop(
        model,
        student_model,
        dataset,
        distil_technique,
        distil_criterion,
        optimizer,
        epochs=epochs,
        threshold=performance_target,
    )
    plot.print_header()

    return compressed_model
