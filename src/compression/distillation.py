import copy
from enum import Enum
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import src.compression.pruning as prune
import general
import plot
from src.interfaces.dataset_models import DataSet
from torch.utils.tensorboard import SummaryWriter

from src.interfaces.techniques import DistillationTechnique


def soft_target_distillation(teacher, student, dataset, distil_criterion, optimizer, device):
    teacher.eval()  # Set teacher model to evaluation mode
    student.train()  # Set student model to training mode

    teacher.to(device)
    student.to(device)

    epoch_loss = 0
    epoch_score = 0

    for data, target in tqdm(dataset.train_loader, desc="Distillation Training", position=0, leave=True, dynamic_ncols=True):
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
        distill_loss = distil_criterion(
            student_output, teacher_output.detach())
        distill_loss.backward()

        # Update the student model's parameters
        optimizer.step()

        # For logging purposes
        epoch_loss += distill_loss.item()
        epoch_score += dataset.metric(student_output, target)

    epoch_loss /= len(dataset.train_loader)
    epoch_score /= len(dataset.train_loader)

    return {"loss": epoch_loss, "score": epoch_score}


def hard_target_distillation(teacher, student, dataset, distil_criterion, optimizer, device):
    teacher.eval()  # Set teacher model to evaluation mode
    student.train()  # Set student model to training mode

    teacher.to(device)
    student.to(device)

    epoch_loss = 0
    epoch_score = 0

    for data, target in tqdm(dataset.train_loader, desc="Distillation Training", position=0, leave=True, dynamic_ncols=True):
        # Move data to the appropriate device
        data = data.to(device)

        # Compute the hard target labels from the teacher model
        with torch.no_grad():
            teacher_output = teacher(data)
            _, hard_target_labels = torch.max(teacher_output, dim=1)

        # Compute the output probabilities of the student model
        student_output = student(data)

        # Zero the gradients before computing the loss
        optimizer.zero_grad()

        # Compute the loss and gradient
        distill_loss = distil_criterion(
            student_output, hard_target_labels.detach())
        distill_loss.backward()

        # Update the student model's parameters
        optimizer.step()

        # For logging purposes
        epoch_loss += distill_loss.item()
        epoch_score += dataset.metric(student_output, target)

    epoch_loss /= len(dataset.train_loader)
    epoch_score /= len(dataset.train_loader)

    return {"loss": epoch_loss, "score": epoch_score}


def get_distil_method(technique):
    if technique == DistillationTechnique.SoftTarget:
        return soft_target_distillation
    elif technique == DistillationTechnique.HardTarget:
        return hard_target_distillation
    elif technique == DistillationTechnique.CombinedLoss:
        return combined_loss_distillation
    else:
        raise ValueError("Invalid distillation technique")


def combined_loss_distillation(teacher, student, dataset, distil_criterion, optimizer, device, alpha=0.2, temperature=1.5):

    teacher.to(device).eval()
    student.to(device).train()

    task_criterion = dataset.criterion

    epoch_loss = 0
    epoch_score = 0

    for data, hard_target in tqdm(dataset.train_loader, desc="Distillation Training", position=0, leave=True, dynamic_ncols=True):
        # Move data and hard_target to the appropriate device
        data = data.to(device)
        hard_target = hard_target.to(device)

        # Compute the soft target probabilities from the teacher model
        with torch.no_grad():
            teacher_output = teacher(data)
            teacher_soft_target = F.softmax(
                teacher_output / temperature, dim=1)

        # Compute the output probabilities of the student model
        student_output = student(data)
        student_soft_output = F.softmax(student_output / temperature, dim=1)

        # Zero the gradients before computing the loss
        optimizer.zero_grad()

        # Compute the task loss (e.g., cross-entropy loss for classification tasks)
        task_loss = task_criterion(student_output, hard_target)

        # Compute the distillation loss (e.g., KL Divergence for soft-target distillation)
        distill_loss = distil_criterion(
            student_soft_output, teacher_soft_target.detach())

        # Combine the task loss and distillation loss using the weight alpha
        combined_loss = (1 - alpha) * task_loss + alpha * distill_loss
        combined_loss.backward()

        # Update the student model's parameters
        optimizer.step()

        # For logging purposes
        epoch_loss += combined_loss.item()
        epoch_score += dataset.metric(student_output, hard_target)

    epoch_loss /= len(dataset.train_loader)
    epoch_score /= len(dataset.train_loader)

    return {"loss": epoch_loss, "score": epoch_score}


# Method that trains the student model using distillation
def distillation_train_loop(
    teacher,
    student,
    dataset: DataSet,
    distil_technique,
    distil_criterion,
    optimizer,
    epochs=1,
    patience=3,
    target=None,
    save_path=None,
    writer=None,
    device=None,
):
    if device is None:
        device = general.get_device()

    teacher.to(device)
    teacher.eval()
    student.to(device)

    # If a threshold is specified, train the student model until the threshold is reached or the score decreases
    if target is not None:
        best_score = 0
        best_model = copy.deepcopy(student)
        epochs_without_improvement = 0

        # Validate the student model before training
        start_metrics = general.validate(student, dataset, device=device)
        score = start_metrics[1]
        logging.info('Start Score: {}'.format(score))

        it = 0

        if writer is not None:
            metrics = {
                "loss": start_metrics[0],
                "score": start_metrics[1],
            }
            plot.log_metrics("distillation", "train", metrics, it, writer)
            plot.log_metrics("distillation", "validation", metrics, it, writer)

        while score < target and epochs_without_improvement < patience:

            it += 1

            # Train the student model
            distil_metrics = distil_technique(
                teacher, student, dataset, distil_criterion, optimizer, device)

            # Validate the student model
            val_metrics = general.validate(student, dataset, device=device)

            if writer is not None:
                plot.log_metrics("distillation", "train",
                                 distil_metrics, it, writer)
                plot.log_metrics("distillation", "validation", {
                    "loss": val_metrics[0],
                    "score": val_metrics[1],
                }, it, writer)

            distil_score = distil_metrics["score"]
            score = val_metrics[1]

            logging.info('Distillation Score: {:.3f}; Validation Score: {:.3f}'.format(
                distil_score, score))

            # If the score is above the threshold, stop training
            if score > target:
                logging.info("Stopped training because target ({}) was reached: {:.3f}".format(
                    target, score))
                break

            # If the score is decreasing, stop training
            if score > best_score:
                best_model = copy.deepcopy(student)
                if save_path is not None:
                    torch.save(best_model, save_path)
                best_score = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logging.info("Stopped training because score did not improve for {} epochs".format(
                patience))

        student = best_model

    # Otherwise, train the student model for the specified number of epochs
    else:
        for e in range(epochs):
            # Train the student model
            distil_technique(teacher, student, dataset,
                             distil_criterion, optimizer)

            # Validate the student model
            metrics = general.validate(student, dataset)
            score = metrics[1]
            logging.info('Score: {}'.format(score))
        logging.info("Stopped training after {} epochs".format(epochs))

    return student


# Method that creates a student model based on the teacher model
def create_student_model(teacher_model, dataset: DataSet, fineTune=True):
    teacher_model = copy.deepcopy(teacher_model)
    prune.channel_pruning(
        teacher_model, dataset, sparsity=0.5, fineTune=fineTune)
    return teacher_model


# Method that performs the whole distillation procedure
def perform_distillation(model, dataset: DataSet, technique=DistillationTechnique.CombinedLoss, student_model=None,  settings: dict = {}, writer: SummaryWriter = None, device=None, save_path=None, **kwargs):

    # Extract settings
    # TODO: Find intelligent way to set the following properties
    performance_target = settings.get("performance_target", None)
    distil_criterion = settings.get("distil_criterion", F.cross_entropy)
    optimizer = settings.get("optimizer", optim.SGD(
        student_model.parameters(), lr=0.01, momentum=0.5))
    patience = settings.get("patience", 3)

    distil_method = get_distil_method(technique)

    # Create student model if not provided
    if student_model is None:
        plot.print_header("Creating student model")
        student_model = create_student_model(model, dataset)

    compressed_model = distillation_train_loop(
        model,
        student_model,
        dataset,
        distil_method,
        distil_criterion,
        optimizer,
        target=performance_target,
        patience=patience,
        writer=writer,
        device=device,
        save_path=save_path
    )

    return compressed_model
