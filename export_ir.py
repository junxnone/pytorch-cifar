import torch
import subprocess


def execute_mo(input_model, output_dir, name, data_type):
    command = [
        'mo',
        '--input_model={}'.format(input_model),
        '--output_dir={}'.format(output_dir),
        '--model_name={}'.format(name),
        '--data_type={}'.format(data_type),
    ]
    subprocess.call(command)

if __name__ == '__main__':

    model_file = "output/ckpt.pth"
    model  = torch.load(model_file, map_location='cpu')

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, (dummy_input, ), 'output/model.onnx')

    execute_mo('output/model.onnx', 'output/FP32/1', 'cifar', 'FP32')