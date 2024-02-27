window.onload = function () {

  // Illustrate NN
  num_layers = 3;
  for (let i = 0; i < num_layers; i++) {
    neurons = "";
    for (let j = 0; j < 8; j++) {
      neurons += `<span id="n-${j}" class="dot"></span>`;
    }

    $("#hidden")
      .append(`<div id="layer_${i}" class="layer" style="text-align: center">
        ${neurons}
    </div>`);
  }

  for (let i = 0; i < 10; i++) {
    $("#output").append(`<span class="dot">${i}</span>`);
  }

  // Handle Canvas
  var myCanvas = document.getElementById("myCanvas");
  var ctx = myCanvas.getContext("2d");



  // Mouse Event Handlers
  if (myCanvas) {
    var isDown = false;
    var canvasX, canvasY;
    ctx.lineWidth = 5;

    $(myCanvas)
      .mousedown(function (e) {
        isDown = true;
        ctx.beginPath();
        canvasX = e.pageX - myCanvas.offsetLeft;
        canvasY = e.pageY - myCanvas.offsetTop;
        ctx.moveTo(canvasX, canvasY);
      })
      .mousemove(function (e) {
        if (isDown !== false) {
          canvasX = e.pageX - myCanvas.offsetLeft;
          canvasY = e.pageY - myCanvas.offsetTop;
          ctx.lineTo(canvasX, canvasY);
          ctx.strokeStyle = "#000";
          ctx.stroke();
        }
      })
      .mouseup(function (e) {
        isDown = false;
        ctx.closePath();
      });
  }

  // Touch Events Handlers
  draw = {
    started: false,
    start: function (evt) {
      ctx.beginPath();
      ctx.moveTo(evt.touches[0].pageX, evt.touches[0].pageY);

      this.started = true;
    },
    move: function (evt) {
      if (this.started) {
        ctx.lineTo(evt.touches[0].pageX, evt.touches[0].pageY);

        ctx.strokeStyle = "#000";
        ctx.lineWidth = 5;
        ctx.stroke();
      }
    },
    end: function (evt) {
      this.started = false;
    },
  };

  // Touch Events
  myCanvas.addEventListener("touchstart", draw.start, false);
  myCanvas.addEventListener("touchend", draw.end, false);
  myCanvas.addEventListener("touchmove", draw.move, false);

  // Disable Page Move
  document.body.addEventListener(
    "touchmove",
    function (evt) {
      evt.preventDefault();
    },
    false
  );

  $("#run").on("click", ()=>{
    let c1 = document.createElement("canvas");
    let ctx1 = c1.getContext('2d')
    c1.width = 28
    c1.height = 28
    ctx1.drawImage(myCanvas, 4, 4, 20, 20);
    // document.getElementById('img').src = c1.toDataURL();
    // document.getElementById('c').style.display = 'none';
    hidden = true
  
    var imgData = ctx1.getImageData(0, 0, 28, 28);
    var imgBlack = []
    for (var i = 0; i < imgData.data.length; i += 4) {
      if (imgData.data[i + 3] === 255){
        imgBlack.push(1)
      }
      else imgBlack.push(0)
    }

    imgVector = nj.array(imgBlack);
  
    const layer_1_out = nj.dot(imgVector, layer_1.T)

    // RELU activation

    const layer_1_out_relu = layer_1_out.tolist().map(v => (v>0)? v:0);
    console.log(layer_1_out_relu);
    const layer_2_out = nj.dot(layer_2, nj.array(layer_1_out_relu));
    
    const layer_2_out_relu = layer_2_out.tolist().map(v => (v>0)? v:0);
    console.log(layer_2_out_relu);
    const layer_3_out = nj.dot(layer_3, nj.array(layer_2_out_relu));
    
    const layer_3_out_relu = layer_3_out.tolist().map(v => (v>0)? v:0);
    console.log(layer_3_out_relu);
    const layer_4_out = nj.dot(nj.array(layer_3_out_relu), layer_4);
    let layer_4_softmax = nj.exp(layer_4_out);
    // console.log(layer_4_softmax.tolist());
    // console.log(layer_4_softmax.sum());
    layer_4_softmax = layer_4_softmax.divide(layer_4_softmax.sum());
    
    layer_4_softmax = layer_4_softmax.tolist()
    console.log(layer_4_softmax);
    console.log(layer_4_softmax.indexOf(Math.max(...layer_4_softmax)));
    // console.log(layer_4_out.tolist());

    // console.log(dataStr)

  });

};
