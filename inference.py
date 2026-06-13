from ultralytics import YOLO
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='test image path')
    parser.add_argument('--weights', type=str, required=True, help='pretrained weights path')
    parser.add_argument('--output', type=str, default='result.jpg', help='output image path')
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model(args.image)
    results[0].save(filename=args.output)
    print(f'Saved result to {args.output}')

if __name__ == '__main__':
    main()
