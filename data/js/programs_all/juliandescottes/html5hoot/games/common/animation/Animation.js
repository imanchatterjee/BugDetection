// Generated by CoffeeScript 1.3.1
(function() {

  Aria.classDefinition({
    $classpath: 'games.common.animation.Animation',
    $extends: 'games.common.BaseObject',
    $constructor: function(image, frameWidth, frameDuration, scale) {
      this.$BaseObject.constructor.call(this);
      this.__image = image;
      this.__frameWidth = frameWidth;
      this.__frameDuration = frameDuration;
      this.__frameHeight = image.height;
      this.__totalTime = (image.width / frameWidth) * frameDuration;
      this.__elapsedTime = 0;
      return this.scale = scale || 1;
    },
    $prototype: {
      update: function(deltaTime) {
        return this.__elapsedTime += deltaTime;
      },
      draw: function(context, x, y) {
        var index, locX, locY;
        index = this.getCurrentFrame();
        locX = x - (this.__frameWidth / 2) * this.scale;
        locY = y - (this.__frameHeight / 2) * this.scale;
        return context.drawImage(this.__image, index * this.__frameWidth, 0, this.__frameWidth, this.__frameHeight, locX, locY, this.__frameWidth * this.scale, this.__frameHeight * this.scale);
      },
      isFinished: function() {
        return this.__elapsedTime >= this.__totalTime;
      },
      getCurrentFrame: function() {
        return Math.floor(this.__elapsedTime / this.__frameDuration);
      }
    }
  });

}).call(this);