import * as ms from '@magenta/sketch';
import p5 from 'p5';

function sketch(p) {
  let modelLoaded = false;
  let dx, dy; // Offsets of the pen strokes, in pixels.
  let x = p.windowWidth / 2.0;
  let y = p.windowHeight / 3.0;
  let pen = [0,0,0]; // Current pen state, [pen_down, pen_up, pen_end].
  let previousPen = [1, 0, 0]; // Previous pen state.
  const PEN = {DOWN: 0, UP: 1, END: 2};
  let modelState;

  const model = new ms.SketchRNN("https://storage.googleapis.com/quickdraw-models/sketchRNN/models/cat.gen.json");

  p.setup = () => {
    const containerSize = document.getElementById('sketch').getBoundingClientRect();
    // Initialize the canvas.
    const screenWidth = Math.floor(containerSize.width);
    const screenHeight = p.windowHeight / 2;
    p.createCanvas(screenWidth, screenHeight);
    p.frameRate(60);

    model.initialize().then(() => {
      modelLoaded = true;
      model.setPixelFactor(3.0);

      [dx, dy, ...pen] = model.zeroInput();
      modelState = model.zeroState();
    });
  }

  p.draw = () => {
    if (!modelLoaded) {
      return;
    }

    modelState = model.update([dx, dy, ...pen], modelState);
    const pdf = model.getPDF(modelState, 0.45);
    [dx, dy, ...pen] = model.sample(pdf);

    if (previousPen[PEN.DOWN] == 1) {
      p.line(x, y, x+dx, y+dy);
    }

    x += dx;
    y += dy;

    previousPen = pen;
  }
}

new p5(sketch, 'SketchRNN');

