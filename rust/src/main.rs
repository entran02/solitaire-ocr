use fantoccini::{ClientBuilder, Locator};
use std::{process::{Child, Command}, time::Duration, fs};
use tokio::time::sleep;
use opencv::core::{Mat, Point, Scalar, Rect};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, match_template, rectangle, LINE_8, TM_CCOEFF_NORMED,COLOR_BGR2GRAY};
use opencv::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone)]
struct BoundingBox {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    label: String,
}

#[derive(Serialize)]
struct GameState {
    draw_pile: Vec<String>,
    game_piles: Vec<Vec<String>>,
    discard_pile: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), fantoccini::error::CmdError> {
    // start chrome and go to solitaire
    let mut chrome = start_chrome()?;

    let client = ClientBuilder::native()
        .connect("http://localhost:4444")
        .await
        .expect("failed to connect to WebDriver");

        
    client.goto("https://www.google.com/logos/fnbx/solitaire/standalone.html").await?;

    client.wait().for_element(Locator::Id("solitaire-easy-button")).await?;
    let easy_btn = client.find(Locator::Id("solitaire-easy-button")).await?;
    easy_btn.click().await?;

    sleep(Duration::from_secs(3)).await;

    // take screenshot
    let ss = client.screenshot().await?;
    std::fs::write("screenshot.png", ss).expect("failed to write screenshot");

    chrome.kill()?;
    chrome.wait()?;

    // convert screenshot to game state
    translate().expect("Failed to translate");

    Ok(())
}

fn translate() -> opencv::Result<()> {
    // to test with manual pngs replace screenshot with png name and comment out the chromium code
    let image_path = "screenshot.png";
    let mut img = load_image(image_path)?;
    let output_path = "output_with_boxes.png";

    let templates = get_templates();
    let card_threshold = 0.79;
    let suit_threshold = 0.85;

    let mut card_bounding_boxes = Vec::new();
    let mut suit_bounding_boxes = Vec::new();

    for template_path in &templates {
        let template = load_image(template_path)?;
        // use png name for label
        let label = template_path.split('\\').last().unwrap().replace(".png", "");

        // match card values and suits with different thresholds for accuracy
        let threshold = if label == "hearts" || label == "diamonds" || label == "clubs" || label == "spades" {
            suit_threshold
        } else {
            card_threshold
        };

        let matches = match_template_with_threshold(&img, &template, threshold)?;
        let boxes = create_bounding_boxes(matches, template.cols(), template.rows(), label);

        if threshold == suit_threshold {
            suit_bounding_boxes.extend(boxes);
        } else {
            card_bounding_boxes.extend(boxes);
        }
    }

    // nms for both
    let filtered_cards = non_maximum_suppression(card_bounding_boxes.clone(), 0.5);
    let filtered_suits = non_maximum_suppression(suit_bounding_boxes.clone(), 0.5);

    draw_bounding_boxes(&mut img, &filtered_cards)?;
    draw_bounding_boxes(&mut img, &filtered_suits)?;

    // save image with bounding boxes
    save_image(&img, output_path)?;

    let game_state = generate_game_state(filtered_cards, filtered_suits, img.cols(), 40);
    let _ = save_game_state(&game_state, "output.json");

    println!("Game state saved to output.json");

    Ok(())
}

fn start_chrome() -> Result<Child, std::io::Error> {
    Command::new("chromedriver")
        .arg("--port=4444")
        .spawn()
}

fn get_templates() -> Vec<String> {
    let template_dir = "templates";
    fs::read_dir(template_dir)
        .expect("Failed to read templates directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().to_str().unwrap().to_string())
        .collect()
}

// load image in greyscale
fn load_image(path: &str) -> opencv::Result<Mat> {
    let img = imread(path, IMREAD_COLOR)?;
    let mut gray = Mat::default();
    cvt_color(&img, &mut gray, COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn match_template_with_threshold(
    img: &Mat,
    template: &Mat,
    threshold: f32,
) -> opencv::Result<Vec<Point>> {
    let mut result = Mat::default();
    // find matches
    match_template(img, template, &mut result, TM_CCOEFF_NORMED, &Mat::default())?;

    // filter matches by threshold
    let mut matches = Vec::new();
    for y in 0..result.rows() {
        for x in 0..result.cols() {
            let value = *result.at_2d::<f32>(y, x)?;
            if value >= threshold {
                matches.push(Point::new(x, y));
            }
        }
    }
    Ok(matches)
}

fn create_bounding_boxes(
    matches: Vec<Point>,
    template_width: i32,
    template_height: i32,
    label: String,
) -> Vec<BoundingBox> {
    matches
        .into_iter()
        .map(|pt| BoundingBox {
            x1: pt.x,
            y1: pt.y,
            x2: pt.x + template_width,
            y2: pt.y + template_height,
            label: label.clone(),
        })
        .collect()
}

fn non_maximum_suppression(
    boxes: Vec<BoundingBox>,
    overlap_thresh: f32,
) -> Vec<BoundingBox> {
    let mut filtered_boxes = Vec::new();
    let mut boxes = boxes.clone();

    // sorted by bottom right corner
    boxes.sort_by(|a, b| b.y2.cmp(&a.y2));
    while let Some(current) = boxes.pop() {
        filtered_boxes.push(current.clone());
        boxes.retain(|b| {
            let inter_x1 = current.x1.max(b.x1);
            let inter_y1 = current.y1.max(b.y1);
            let inter_x2 = current.x2.min(b.x2);
            let inter_y2 = current.y2.min(b.y2);

            let inter_area = (inter_x2 - inter_x1).max(0) * (inter_y2 - inter_y1).max(0);
            let box_area = (b.x2 - b.x1) * (b.y2 - b.y1);
            let overlap = inter_area as f32 / box_area as f32;

            overlap <= overlap_thresh
        });
    }
    filtered_boxes
}

fn associate_cards_and_suits(
    cards: Vec<BoundingBox>,
    suits: Vec<BoundingBox>,
) -> Vec<BoundingBox> {
    let mut associated_cards = Vec::new();

    for mut card in cards {
        let mut closest_suit = None;
        let mut min_distance = i32::MAX;

        // find closest suit
        for suit in &suits {
            let horizontal_distance = (suit.x1 - card.x2).abs();
            let vertical_overlap = (suit.y1 <= card.y2) && (suit.y2 >= card.y1);

            if vertical_overlap && horizontal_distance < min_distance {
                min_distance = horizontal_distance;
                closest_suit = Some(suit.label.clone());
            }
        }

        // associate card with suit
        if let Some(suit_label) = closest_suit {
            card.label = format!("{} {}", card.label, suit_label);
        }

        associated_cards.push(card);
    }

    associated_cards
}


fn group_bounding_boxes_by_x_percentage(
    bounding_boxes: &[BoundingBox],
    image_width: i32,
) -> std::collections::HashMap<String, Vec<BoundingBox>> {
    let percentage_ranges = (0..9).map(|i| (i as f32 / 9.0, (i + 1) as f32 / 9.0));
    let mut grouped_boxes: std::collections::HashMap<String, Vec<BoundingBox>> = 
        percentage_ranges.clone().map(|(start, end)| (format!("{:.0}%-{:.0}%", start * 100.0, end * 100.0), Vec::new()))
        .collect();

    for b in bounding_boxes {
        let center_x = (b.x1 + b.x2) as f32 / 2.0;
        let x_percentage = center_x / image_width as f32;

        for (start, end) in percentage_ranges.clone() {
            if start <= x_percentage && x_percentage < end {
                let range_key = format!("{:.0}%-{:.0}%", start * 100.0, end * 100.0);
                grouped_boxes.get_mut(&range_key).unwrap().push(b.clone());
                break;
            }
        }
    }

    grouped_boxes
}

fn group_bounding_boxes_by_y_range(
    bounding_boxes: &[BoundingBox],
    y_range_step: i32,
) -> Vec<Vec<BoundingBox>> {
    let mut grouped_rows: Vec<Vec<BoundingBox>> = Vec::new();
    let mut current_row: Vec<BoundingBox> = Vec::new();

    let mut y_start = 0;
    let mut y_end = y_range_step;

    for b in bounding_boxes {
        let center_y = (b.y1 + b.y2) / 2;

        if y_start <= center_y && center_y < y_end {
            current_row.push(b.clone());
        } else {
            if !current_row.is_empty() {
                grouped_rows.push(current_row.clone());
            }
            current_row = vec![b.clone()];
            y_start = center_y / y_range_step * y_range_step;
            y_end = y_start + y_range_step;
        }
    }

    if !current_row.is_empty() {
        grouped_rows.push(current_row);
    }

    grouped_rows
}

fn generate_game_state(
    cards: Vec<BoundingBox>,
    suits: Vec<BoundingBox>,
    image_width: i32,
    y_range_step: i32,
) -> GameState {
    let associated_cards = associate_cards_and_suits(cards, suits);

    let grouped_by_x = group_bounding_boxes_by_x_percentage(&associated_cards, image_width);

    let mut draw_pile = Vec::new();
    let mut game_piles = vec![Vec::new(); 7];
    let mut discard_pile = vec![None; 4];

    for (x_range, boxes) in grouped_by_x {
        let rows = group_bounding_boxes_by_y_range(&boxes, y_range_step);

        if x_range == "0%-11%" {
            draw_pile = rows
                .iter()
                .flat_map(|row| row.iter().map(|b| b.label.clone()))
                .collect();
        } else if x_range == "89%-100%" {
            for (i, row) in rows.iter().enumerate().take(4) {
                if let Some(b) = row.first() {
                    // temp: filters out J from discard, for some reason its always matched in that area
                    if b.label.contains("J") {
                        discard_pile[i] = Some("null".to_string());
                    } else {
                        discard_pile[i] = Some(b.label.clone());
                    }
                }
            }
        } else if let Ok(start_percentage) = x_range
            .split('-')
            .next()
            .unwrap()
            .trim_end_matches('%')
            .parse::<i32>()
        {
            let index = ((start_percentage - 11) / 11) as usize;
            if index < game_piles.len() {
                let starting_y = 75;

                if let Some(first_box) = rows
                .iter()
                .flat_map(|row| row.iter())
                .min_by_key(|b| b.y1)
                {
                    let null_rows = (first_box.y1.saturating_sub(starting_y)) / y_range_step;
                    game_piles[index].resize(null_rows as usize, "null".to_string());
                }
                for row in rows {
                    game_piles[index].extend(row.iter().map(|b| b.label.clone()));
                }
            }
        }
    }

    let discard_pile: Vec<String> = discard_pile
        .into_iter()
        .map(|card| card.unwrap_or_else(|| "null".to_string()))
        .collect();

    GameState {
        draw_pile,
        game_piles,
        discard_pile,
    }
}

fn save_game_state(state: &GameState, path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(state)?;
    fs::write(path, json)?;
    Ok(())
}

fn draw_bounding_boxes(img: &mut Mat, bounding_boxes: &[BoundingBox]) -> opencv::Result<()> {
    for bounding_box in bounding_boxes {
        let rect = Rect::new(
            bounding_box.x1,
            bounding_box.y1,
            bounding_box.x2 - bounding_box.x1,
            bounding_box.y2 - bounding_box.y1,
        );

        rectangle(
            img,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            LINE_8,
            0,
        )?;
    }
    Ok(())
}

fn save_image(img: &Mat, output_path: &str) -> opencv::Result<()> {
    imwrite(output_path, img, &opencv::core::Vector::new())
        .and_then(|success| if success {
            Ok(())
        } else {
            Err(opencv::Error::new(opencv::core::StsError, "Failed to save image"))
        })
}
