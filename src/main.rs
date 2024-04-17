use std::time::Duration;
use rand::prelude::*;
use halton::Sequence;
use bevy::{
    math::Vec3Swizzles,
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_spatial::{kdtree::KDTree2, SpatialAccess};
use bevy_spatial::{AutomaticUpdate, SpatialStructure};

const BOUNDS: Vec2 = Vec2::new(1300.0, 760.0);
const BOID_COUNT: i32 = 1000;
const BOID_SIZE: f32 = 0.5;
const BOID_SPEED: f32 = 300.;
const BOID_VIS_RANGE: f32 = 50.0;
const VIS_RANGE_SQ: f32 = BOID_VIS_RANGE*BOID_VIS_RANGE;
const BOID_PROT_RANGE: f32 = 10.0;
const PROT_RANGE_SQ: f32 = BOID_PROT_RANGE*BOID_PROT_RANGE;
const BOID_CENTER_FACTOR: f32 = 0.0005;
const BOID_MATCHING_FACTOR: f32 = 0.05;
const BOID_AVOID_FACTOR: f32 = 0.05;
const BOID_MIN_SPEED: f32 = 50.0;
const BOID_MAX_SPEED: f32 = 200.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Track boids in the KD-Tree
        .add_plugins(
            AutomaticUpdate::<SpatialEntity>::new()
                // TODO: check perf of other tree types
                .with_spatial_ds(SpatialStructure::KDTree2)
                .with_frequency(Duration::from_millis(16)),
        )
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, (update_position))
        .add_systems(Update, bevy::window::close_on_esc)
        .run();
}

// Marker for entities tracked by KDTree
#[derive(Component)]
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
        let spawn_x = (x as f32 * BOUNDS.x) - BOUNDS.x / 2.0;
        let spawn_y = (y as f32 * BOUNDS.y) - BOUNDS.y / 2.0;

        let mut transform = Transform::from_xyz(spawn_x, spawn_y, 0.0)
            .with_scale(Vec3::splat(BOID_SIZE));

        transform.rotate_z(0.0);

        let velocity = Velocity(Vec2::new(rng.gen_range(-1.0..1.0),
                                          rng.gen_range(-1.0..1.0)) * BOID_SPEED);

        commands.spawn((
            BoidBundle {
                mesh: MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(meshes.add(Circle { radius: 4.0 })),
                    material: materials.add(
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

// fn update_velocity(
//     tree: Res<KDTree2<SpatialEntity>>,
//     qboid: Query<(Entity, &Transform), With<SpatialEntity>>,
//     mut qvelocity: Query<&mut Velocity>
// ) {
//     for (boid, t0) in qboid.iter() {
//         let mut xpos_avg = 0.;
//         let mut ypos_avg = 0.;
//         let mut xvel_avg = 0.;
//         let mut yvel_avg = 0.;
//         let mut close_dx = 0.;
//         let mut close_dy = 0.;
//         let mut neighboring_boids = 0.;
//
//         let p0 = t0.translation.xy();
//
//         let Ok(mut v0) = qvelocity.get_mut(boid) else { todo!(); };
//
//         for (_, entity) in tree.within_distance(p0, BOID_VIS_RANGE) {
//             let Ok((_, t1)) = qboid.get(entity.unwrap()) else { todo!() };
//             let Ok(v1) = qvelocity.get(entity.unwrap()) else { todo!() };
//
//             // don't evaluate boid against itself
//             if t0.translation == t1.translation {
//                 continue;
//             }
//
//             let p1 = t1.translation.xy();
//
//             let dx = p0.x - p1.x;
//             let dy = p0.y - p1.y;
//
//             let squared_distance = dx*dx + dy*dy;
//
//             if squared_distance < PROT_RANGE_SQ {
//                 close_dx += dx;
//                 close_dy += dy;
//             } else {
//                 xpos_avg += p1.x;
//                 ypos_avg += p1.y;
//                 xvel_avg += v1.0.x;
//                 yvel_avg += v1.0.y;
//                 neighboring_boids += 1.;
//             }
//         }
//
//         if neighboring_boids > 0. {
//             xpos_avg /= neighboring_boids;
//             ypos_avg /= neighboring_boids;
//             xvel_avg /= neighboring_boids;
//             yvel_avg /= neighboring_boids;
//
//             v0.0.x += (xpos_avg - p0.x) * BOID_CENTER_FACTOR + (xvel_avg - p0.x) * BOID_MATCHING_FACTOR;
//             v0.0.y += (ypos_avg - p0.y) * BOID_CENTER_FACTOR + (yvel_avg - p0.y) * BOID_MATCHING_FACTOR;
//         }
//
//         v0.0.x += close_dx * BOID_AVOID_FACTOR;
//         v0.0.y += close_dy * BOID_AVOID_FACTOR;
//
//         let speed = (v0.0.x*v0.0.x + v0.0.y*v0.0.y).sqrt();
//
//         if speed < BOID_MIN_SPEED {
//             v0.0.x *= BOID_MIN_SPEED / speed;
//             v0.0.y *= BOID_MIN_SPEED / speed;
//         }
//         if speed > BOID_MAX_SPEED {
//             v0.0.x *= BOID_MAX_SPEED / speed;
//             v0.0.y *= BOID_MAX_SPEED / speed;
//         }
//     }
// }

fn update_position(
    time: Res<Time>,
    mut query: Query<(&mut Velocity, &mut Transform)>,
) {
    for (mut velocity, mut transform) in query.iter_mut() {
        if transform.translation.x.abs() < BOUNDS.x / 2.0
            && transform.translation.y.abs() < BOUNDS.y / 2.0
        {
            transform.translation += Vec3::from((velocity.0, 0.0)) * time.delta_seconds();
        } else {
            velocity.0 *= -1.0;
            transform.translation += Vec3::from((velocity.0, 0.0)) * time.delta_seconds();
        }
    }
}