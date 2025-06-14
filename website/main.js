window.onload = function () {

  // Illustrate NN
  num_layers = 3;
  for (let i = 0; i < num_layers; i++) {
    neurons = "";
    for (let j = 0; j < 10; j++) {
      neurons += `<span id="n-${j}" class="dot"></span>`;
    }

    $("#hidden")
      .append(`<div id="layer_${i}" class="layer" style="text-align: center">
        ${neurons}
    </div>`);
  }

  for (let i = 0; i < 10; i++) {
    $("#output").append(`<span id="n-${i}" class="dot">${i}</span>`);
  }

  var myCanvas = document.getElementById("myCanvas");
  if (myCanvas) {
    var ctx = myCanvas.getContext("2d");

    // Set the entire canvas to black
    ctx.fillStyle = "#000"; // Set fill color to black
    ctx.fillRect(0, 0, myCanvas.width, myCanvas.height); // Fill the canvas with black

    // Set drawing parameters for white lines
    ctx.strokeStyle = "#FFF"; // Set drawing color to white
    ctx.lineWidth = 25; // You can adjust lineWidth as needed
    ctx.lineCap = 'round';

    // Your existing mouse and touch event handlers...

  // Touch Events Handlers
  draw = {
    started: false,
    start: function (evt) {
        ctx.beginPath();
        ctx.moveTo(evt.clientX, evt.clientY);
        ctx.strokeStyle = "#FFF";
        ctx.lineWidth = 25;
        this.started = true;
    },
    move: function (evt) {
        if (this.started) {
            ctx.lineTo(evt.clientX, evt.clientY);
            ctx.strokeStyle = "#FFF";
            ctx.lineWidth = 25;
            ctx.stroke();
        }
    },
    end: function (evt) {
        this.started = false;
    }
};

myCanvas.addEventListener('mousedown', draw.start.bind(draw));
myCanvas.addEventListener('mousemove', draw.move.bind(draw));
myCanvas.addEventListener('mouseup', draw.end.bind(draw));

  $("#clear").on("click", function(){
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, myCanvas.width, myCanvas.height);
    $(".dot").css({"background-color": "#bbb", "color": "black"});
  });
  }

    function resizeCanvasImage(canvas, width, height) {
        // Create a temporary canvas to draw the resized image
        var tempCanvas = document.createElement('canvas');
        var tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = width;
        tempCanvas.height = height;

        // Draw the original canvas onto the temporary canvas with resizing
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, width, height);

        return tempCanvas;
    }

    function convertToGrayscale(canvas) {
        var ctx = canvas.getContext('2d');
        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        var data = imageData.data;
        
        for (var i = 0; i < data.length; i += 4) {
            // Convert the pixel to grayscale using luminosity method
            var grayscale = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            data[i] = data[i + 1] = data[i + 2] = grayscale;
        }

        ctx.putImageData(imageData, 0, 0);
    }


    function flattenImage(canvas) {
        var ctx = canvas.getContext('2d');
        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        var data = imageData.data;
        var grayscaleVector = [];

        for (var i = 0; i < data.length; i += 4) {
            grayscaleVector.push(data[i]); // Since it's grayscale, R=G=B
        }

        return grayscaleVector;
    }
    function normalizeVector(vector) {
        return vector.map(value => value / 255);
    }

    function delay(time) {
        return new Promise(resolve => setTimeout(resolve, time));
    }

    async function animate_layer(layer_num, layer_output){
        for (let i = 0; i < layer_output.length; i++){
            if (layer_output[i] > 0) {
                $(`#layer_${layer_num} > #n-${i}`).css('background-color', 'black');
                $(`#layer_${layer_num} > #n-${i}`).css('color', 'white');
                await delay(100);
                // $(`#layer_${layer_num} > #n-${i}`).append('', `${layer_output[i]}`);

            }     
        } 
    }



  $("#run").on("click", async ()=>{
    var resizedCanvas = resizeCanvasImage(myCanvas, 28, 28);
    convertToGrayscale(resizedCanvas);
    var flattenedVector = flattenImage(resizedCanvas);
    var normalizedVector = normalizeVector(flattenedVector);

    imgVector = nj.array(normalizedVector);
    let layer_1_out = nj.dot(layer_1.T, imgVector);
    layer_1_out = nj.array(layer_1_out.tolist().map(v => (v > 0.0)? v:0.0)); // ReLU
    // Animate Neurons For Layer 1
    await animate_layer(0, layer_1_out.tolist());
    await delay(500);
    let layer_2_out = nj.dot(layer_2.T, layer_1_out);
    layer_2_out = nj.array(layer_2_out.tolist().map(v => (v > 0.0)? v:0.0));// ReLU
    // Animate Neurons For layer 2
    await animate_layer(1, layer_2_out.tolist());
    await delay(500);
    let layer_3_out = nj.dot(layer_3.T, layer_2_out);
    layer_3_out = nj.array(layer_3_out.tolist().map(v => (v > 0.0)? v:0.0)); // ReLU
    // Animate Neurons for layer 3
    await animate_layer(2, layer_3_out.tolist());
    await delay(500);
    let layer_4_out = nj.dot(layer_4.T, layer_3_out);
    layer_4_out = nj.exp(layer_4_out);
    layer_4_out = layer_4_out.divide(layer_4_out.sum());
    layer_4_out = layer_4_out.tolist();
    let maxVal = layer_4_out.reduce((maxIndex, currentElement, currentIndex, layer_4_out) => 
    currentElement > layer_4_out[maxIndex] ? currentIndex : maxIndex, 0);

    $(`#output > #n-${maxVal}`).css('background-color', 'black');
    $(`#output > #n-${maxVal}`).css('color', 'white');

  });

};
