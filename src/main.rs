use std::time::Duration;
use rand::prelude::*;
use halton::Sequence;
use bevy::{
    math::Vec3Swizzles,
    prelude::*,
    render::{mesh::*, render_asset::RenderAssetUsages},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    tasks::ComputeTaskPool
};
use bevy_spatial::{
    AutomaticUpdate,
    kdtree::KDTree2,
    SpatialAccess,
    SpatialStructure
};

const WINDOW_BOUNDS: Vec2 = Vec2::new(800., 600.);
const BOID_BOUNDS: Vec2 = Vec2::new(WINDOW_BOUNDS.x * 2./3., WINDOW_BOUNDS.y * 2./3.);
const BOID_COUNT: i32 = 1000;
const BOID_SIZE: f32 = 4.;
const BOID_SPEED: f32 = 100.;
const BOID_VIS_RANGE: f32 = 40.;
const BOID_PROT_RANGE: f32 = 8.;
const BOID_FOV_DEG: f32 = 120.;
const PROT_RANGE_SQ: f32 = BOID_PROT_RANGE*BOID_PROT_RANGE;
const BOID_CENTER_FACTOR: f32 = 0.0005;
const BOID_MATCHING_FACTOR: f32 = 0.05;
const BOID_AVOID_FACTOR: f32 = 0.05;
const BOID_TURN_FACTOR: f32 = 0.2;
const BOID_MOUSE_CHASE_FACTOR: f32 = 0.005;
const BOID_MIN_SPEED: f32 = 2.0;
const BOID_MAX_SPEED: f32 = 4.0;
const BOID_SPEED_DECAY: f32 = 1.;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: (WINDOW_BOUNDS.x, WINDOW_BOUNDS.y).into(),
                   ..default()
                }),
                ..default()
            }),
            // Track boids in the KD-Tree
            AutomaticUpdate::<SpatialEntity>::new()
                // TODO: check perf of other tree types
                .with_spatial_ds(SpatialStructure::KDTree2)
                .with_frequency(Duration::from_millis(16)),
        ))
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .add_event::<DvEvent>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, (
            flocking_system,
            velocity_system,
            movement_system
        ).chain())
        .add_systems(Update, (
            draw_boid_gizmos,
            bevy::window::close_on_esc,
        ))
        .run();
}

// Marker for entities tracked by KDTree
#[derive(Component, Default)]
struct SpatialEntity;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Bundle)]
struct BoidBundle {
    mesh: MaterialMesh2dBundle<ColorMaterial>,
    velocity: Velocity,
}

impl Default for BoidBundle {
    fn default() -> Self {
        Self {
            mesh: Default::default(),
            velocity: Velocity(Vec2::default()),
        }
    }
}

// Event for a change of velocity on some boid
#[derive(Event)]
struct DvEvent(Entity, Vec2);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    let mut rng = rand::thread_rng();

    // Halton sequence for Boid spawns
    let seq = halton::Sequence::new(2).zip(Sequence::new(3))
        .zip(1..BOID_COUNT);

    for ((x, y), _) in seq {
        let spawn_x = (x as f32 * WINDOW_BOUNDS.x) - WINDOW_BOUNDS.x / 2.0;
        let spawn_y = (y as f32 * WINDOW_BOUNDS.y) - WINDOW_BOUNDS.y / 2.0;

        let mut transform = Transform::from_xyz(spawn_x, spawn_y, 0.0)
            .with_scale(Vec3::splat(BOID_SIZE));

        transform.rotate_z(0.0);

        let velocity = Velocity(Vec2::new(rng.gen_range(-1.0..1.0),
                                          rng.gen_range(-1.0..1.0)) * BOID_SPEED);

        commands.spawn((
            BoidBundle {
                mesh: MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(meshes.add(
                        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![
                                [-0.5,  0.5, 0.0],
                                [ 1.0,  0.0, 0.0],
                                [-0.5, -0.5, 0.0],
                                [ 0.0,  0.0, 0.0]
                            ])
                            .with_inserted_indices(Indices::U32(vec![
                                1, 3, 0,
                                1, 2, 3,
                            ]))
                    )),
                    material: materials.add(
                        // Random color for each boid
                        Color::hsl(360. * rng.gen::<f32>(), rng.gen(), 0.7)
                    ),
                    transform,
                    ..default()
                },
                velocity,
            },
            SpatialEntity
        ));
    }
}

fn draw_boid_gizmos(
    mut gizmos: Gizmos,
) {
    gizmos.rect_2d(Vec2::ZERO, 0.0, BOID_BOUNDS, Color::GRAY);
}

fn flocking_dv(
    kdtree: &Res<KDTree2<SpatialEntity>>,
    boid_query: &Query<(Entity, &Velocity, &Transform), With<SpatialEntity>>,
    camera: &Query<(&Camera, &GlobalTransform)>,
    window: &Query<&Window>,
    boid: &Entity,
    t0: &&Transform,
) -> Vec2 {
    // https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
    let mut dv = Vec2::default();
    let mut vec_away = Vec2::default();
    let mut avg_position = Vec2::default();
    let mut avg_velocity = Vec2::default();
    let mut neighboring_boids = 0;
    let mut close_boids = 0;

    for (_, entity) in kdtree.within_distance(t0.translation.xy(), BOID_VIS_RANGE) {
        let Ok((other, v1, t1)) = boid_query.get(entity.unwrap()) else { todo!() };

        // Don't evaluate against itself
        if *boid == other {
            continue;
        }

        // Don't evaluate boids behind
        if t0.forward().angle_between(t1.translation - t0.translation) > f32::to_radians(BOID_FOV_DEG) {
            continue;
        }

        let vec_to = (t1.translation - t0.translation).xy();
        let dist_sq = vec_to.x*vec_to.x + vec_to.y*vec_to.y;

        if dist_sq < PROT_RANGE_SQ {
            // separation
            vec_away -= vec_to;
            close_boids += 1;
        } else {
            // cohesion
            avg_position += vec_to;
            // alignment
            avg_velocity += v1.0;
            neighboring_boids += 1;
        }
    }

    if neighboring_boids > 0 {
        let neighbors = neighboring_boids as f32;
        dv += avg_position / neighbors * BOID_CENTER_FACTOR;
        dv += avg_velocity / neighbors * BOID_MATCHING_FACTOR;
    }

    if close_boids > 0 {
        let close = close_boids as f32;
        dv += vec_away / close * BOID_AVOID_FACTOR;
    }

    // Chase the mouse
    let (camera, t_camera) = camera.single();
    if let Some(c_window) = window.single().cursor_position() {
        if let Some(c_world) = camera.viewport_to_world_2d(t_camera, c_window) {
            let to_cursor = c_world - t0.translation.xy();
            dv += to_cursor * BOID_MOUSE_CHASE_FACTOR;
        } else {};
    } else {};

    dv
}

fn flocking_system(
    boid_query: Query<(Entity, &Velocity, &Transform), With<SpatialEntity>>,
    kdtree: Res<KDTree2<SpatialEntity>>,
    mut dv_event_writer: EventWriter<DvEvent>,
    camera: Query<(&Camera, &GlobalTransform)>,
    window: Query<&Window>,
) {
    let pool = ComputeTaskPool::get();
    let boids = boid_query.iter().collect::<Vec<_>>();
    let boids_per_thread = (boids.len() + pool.thread_num() - 1) / pool.thread_num();

    // https://docs.rs/bevy/latest/bevy/tasks/struct.ComputeTaskPool.html
    for batch in pool.scope(|s| {
        for chunk in boids.chunks(boids_per_thread) {
            let kdtree = &kdtree;
            let boid_query = &boid_query;
            let camera = &camera;
            let window = &window;

            s.spawn(async move {
                let mut dv_batch: Vec<DvEvent> = vec![];

                for (boid, _, t0) in chunk {
                    dv_batch.push(DvEvent(*boid, flocking_dv(
                        kdtree, boid_query, camera, window, boid, t0
                    )));
                }

                dv_batch
            });
        }
    }) {
        dv_event_writer.send_batch(batch);
    }
}

fn velocity_system(
    mut events: EventReader<DvEvent>,
    mut boids: Query<(&mut Velocity, &mut Transform)>,
) {
    for DvEvent(boid, dv) in events.read() {
        let Ok((mut velocity, transform)) = boids.get_mut(*boid) else { todo!() };

        velocity.0.x += dv.x;
        velocity.0.y += dv.y;

        let width = BOID_BOUNDS.x / 2.;
        let height = BOID_BOUNDS.y / 2.;

        // Steer back into visible region
        if transform.translation.x < -width {
            velocity.0.x += BOID_TURN_FACTOR;
        }
        if transform.translation.x > width {
            velocity.0.x -= BOID_TURN_FACTOR;
        }
        if transform.translation.y < -height {
            velocity.0.y += BOID_TURN_FACTOR;
        }
        if transform.translation.y > height {
            velocity.0.y -= BOID_TURN_FACTOR;
        }

        // Clamp speed
        let speed = velocity.0.length();

        if speed < BOID_MIN_SPEED {
            velocity.0 *= BOID_MIN_SPEED / speed;
        }
        if speed > BOID_MAX_SPEED {
            velocity.0 *= BOID_MAX_SPEED / speed;
        }

        // Dampen velocity
        velocity.0 *= BOID_SPEED_DECAY;
    }
}

fn movement_system(
    mut query: Query<(&mut Velocity, &mut Transform)>,
) {
    for (velocity, mut transform) in query.iter_mut() {
        // https://stackoverflow.com/a/68929139
        let angle = velocity.0.y.atan2(velocity.0.x);
        transform.rotation = Quat::from_axis_angle(Vec3::Z, angle);
        transform.translation.x += velocity.0.x;
        transform.translation.y += velocity.0.y;
    }
}
