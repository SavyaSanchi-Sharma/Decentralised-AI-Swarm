use serde::{Deserialize, Serialize};
use crate::model::DynamicModel;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Message {
    Model(DynamicModel),
    ModelUpdate(DynamicModel),
    RequestModel,
}