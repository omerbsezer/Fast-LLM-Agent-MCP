import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Shuttle Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Player (Shuttle) Class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(BLUE) # Simple blue square for shuttle
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.speed = 7

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # Keep player within screen bounds
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top < SCREEN_HEIGHT // 2: # Limit vertical movement to top half
            self.rect.top = SCREEN_HEIGHT // 2
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.top)
        all_sprites.add(bullet)
        bullets.add(bullet)

# Meteor Class
class Meteor(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([30, 30])
        self.image.fill(RED) # Simple red square for meteor
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-100, -40) # Start off-screen above
        self.speed_y = random.randrange(3, 7)

    def update(self):
        self.rect.y += self.speed_y
        if self.rect.top > SCREEN_HEIGHT:
            self.kill() # Remove meteor when it goes off-screen

# Bullet Class
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([5, 15])
        self.image.fill(YELLOW) # Simple yellow rectangle for bullet
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed_y = -10 # Moves upwards

    def update(self):
        self.rect.y += self.speed_y
        if self.rect.bottom < 0:
            self.kill() # Remove bullet when it goes off-screen

# Game Variables
score = 0
font = pygame.font.Font(None, 36) # Default font, size 36

# Sprite Groups
all_sprites = pygame.sprite.Group()
meteors = pygame.sprite.Group()
bullets = pygame.sprite.Group()

# Create player
player = Player()
all_sprites.add(player)

# Game Loop
running = True
clock = pygame.time.Clock()

# Meteor spawning timer
meteor_spawn_event = pygame.USEREVENT + 1
pygame.time.set_timer(meteor_spawn_event, 1000) # Spawn a meteor every 1 second

game_over = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                player.shoot()
        if event.type == meteor_spawn_event and not game_over:
            new_meteor = Meteor()
            all_sprites.add(new_meteor)
            meteors.add(new_meteor)

    if not game_over:
        # Update
        all_sprites.update()

        # Check for collisions: Bullet vs. Meteor
        hits = pygame.sprite.groupcollide(bullets, meteors, True, True)
        for hit in hits:
            score += 10 # Increase score for each meteor hit

        # Check for collisions: Player vs. Meteor
        player_hits = pygame.sprite.spritecollide(player, meteors, False)
        if player_hits:
            game_over = True

    # Drawing
    screen.fill(BLACK) # Background
    all_sprites.draw(screen)

    # Display Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Game Over screen
    if game_over:
        game_over_text = font.render("GAME OVER! Press Q to Quit", True, WHITE)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(game_over_text, text_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

    pygame.display.flip() # Update the display

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
sys.exit()