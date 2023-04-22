import cv2
import snake
import sys
import math





# Load the  image
image = cv2.imread( "pic/tri.jpg", cv2.IMREAD_COLOR )
image=cv2.resize(image, (400, 400))
 

#create the snake
snake = snake.Snake( image, closed = True )

# trackbars
snake_window_name = "Active_contour"
parameters_adjustable_window_name = "parameters_adjustable"
cv2.namedWindow( snake_window_name )
cv2.namedWindow( parameters_adjustable_window_name )
cv2.createTrackbar( "Alpha", parameters_adjustable_window_name, math.floor( snake.alpha * 100 ), 100, snake.set_alpha )
cv2.createTrackbar( "Beta",  parameters_adjustable_window_name, math.floor( snake.beta * 100 ), 100, snake.set_beta )
cv2.createTrackbar( "Delta", parameters_adjustable_window_name, math.floor( snake.delta * 100 ), 100, snake.set_delta )
cv2.createTrackbar( "W Edge", parameters_adjustable_window_name, math.floor( snake.w_edge * 100 ), 100, snake.set_w_edge )
cv2.createTrackbar( "W Line", parameters_adjustable_window_name, math.floor( snake.w_line * 100 ), 100, snake.set_w_line )





while( True ):

    
    snakeImg = snake.visualize()
    # Shows the image
    cv2.imshow( snake_window_name, snakeImg )
    
    snake_changed = snake.energies_minimization_step()
    perimeter=snake.get_length()
    print('perimeter=',perimeter)

    
    k = cv2.waitKey(33)
    if k == 27:
        break


cv2.destroyAllWindows()
