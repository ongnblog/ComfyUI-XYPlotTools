import { app } from "../../scripts/app.js";

const NODE_CONFIG = {
  OGN_XYCheckpointAxis: {
    label: "Checkpoint",
    button: "+ Add Checkpoint",
    rowRemoveButton: "- Remove",
    rows: [{ kind: "combo", prefix: "checkpoint_", source: "checkpoint_1" }],
  },
  OGN_XYDiffusionModelAxis: {
    label: "Diffusion Model",
    button: "+ Add Diffusion Model",
    rowRemoveButton: "- Remove",
    rows: [{ kind: "combo", prefix: "diffusion_model_", source: "diffusion_model_1" }],
  },
  OGN_XYLoraAxis: {
    label: "LoRA",
    loraSets: true,
  },
  OGN_XYPromptSRAxis: {
    label: "Prompt S/R",
    button: "+ Add Replacement",
    removeButton: "- Remove Replacement",
    rows: [{ kind: "string", prefix: "replace_", value: "" }],
  },
};

function getOptions(nodeData, source) {
  const required = nodeData?.input?.required || {};
  const spec = required[source];
  if (Array.isArray(spec?.[0])) {
    return spec[0];
  }
  return ["None"];
}

function nextIndex(node, config) {
  let index = 1;
  for (const widget of node.widgets || []) {
    for (const row of config.rows) {
      if (widget.name?.startsWith(row.prefix)) {
        const value = Number(widget.name.slice(row.prefix.length));
        if (!Number.isNaN(value)) {
          index = Math.max(index, value);
        }
      }
    }
  }
  return index + 1;
}

function moveButtonToEnd(node) {
  if (!node.widgets) {
    return;
  }
  if (node.ognAddButton && !node.ognButtonSpacer) {
    node.ognButtonSpacer = {
      type: "custom",
      name: "ogn_button_spacer",
      value: null,
      serialize: false,
      computeSize: () => [0, 10],
      draw: () => {},
    };
  }
  if (node.ognButtonSpacer) {
    const spacerIndex = node.widgets.indexOf(node.ognButtonSpacer);
    if (spacerIndex >= 0) {
      node.widgets.splice(spacerIndex, 1);
    }
    node.widgets.push(node.ognButtonSpacer);
  }
  for (const button of [node.ognAddButton, node.ognRemoveButton]) {
    if (!button) continue;
    const index = node.widgets.indexOf(button);
    if (index >= 0) {
      node.widgets.splice(index, 1);
    }
    node.widgets.push(button);
  }
}

function addWidget(node, row, index, nodeData, value) {
  const name = `${row.prefix}${index}`;
  if ((node.widgets || []).some((widget) => widget.name === name)) {
    return;
  }
  if (row.kind === "combo") {
    const options = getOptions(nodeData, row.source);
    const selected = options.includes(value) ? value : (value ?? options[0] ?? "None");
    node.addWidget("combo", name, selected, null, { values: options });
  } else if (row.kind === "float") {
    node.addWidget("number", name, value ?? row.value ?? 1.0, null, {
      min: -20,
      max: 20,
      step: 0.05,
      precision: 3,
    });
  } else {
    node.addWidget("text", name, value ?? row.value ?? "", null, {});
  }
}

function removeRow(node, config, rowIndex) {
  for (const widget of [...(node.widgets || [])]) {
    const isRowWidget = config.rows.some((row) => widget.name === `${row.prefix}${rowIndex}`);
    const isRemoveButton = widget.ognRowRemoveIndex === rowIndex;
    const isSpacer = widget.ognRowSpacerIndex === rowIndex;
    if (!isRowWidget && !isRemoveButton && !isSpacer) {
      continue;
    }
    const widgetIndex = node.widgets.indexOf(widget);
    if (widgetIndex >= 0) {
      node.widgets.splice(widgetIndex, 1);
    }
  }
  resizeNode(node);
}

function addRowRemoveButton(node, config, index) {
  if (!config.rowRemoveButton || index <= 1) {
    return;
  }
  if ((node.widgets || []).some((widget) => widget.ognRowRemoveIndex === index)) {
    return;
  }
  const widget = node.addWidget("button", config.rowRemoveButton, null, () => {
    removeRow(node, config, index);
  });
  widget.ognRowRemoveIndex = index;
}

function addRowSpacer(node, config, index) {
  if (!config.rowRemoveButton || index <= 1) {
    return;
  }
  if ((node.widgets || []).some((widget) => widget.ognRowSpacerIndex === index)) {
    return;
  }
  const widget = {
    type: "custom",
    name: `ogn_spacer_${config.label}_${index}`,
    value: null,
    serialize: false,
    ognRowSpacerIndex: index,
    computeSize: () => [0, 10],
    draw: () => {},
  };
  node.widgets.push(widget);
}

function addRow(node, config, nodeData, values = null) {
  const index = nextIndex(node, config);
  addRowSpacer(node, config, index);
  config.rows.forEach((row, rowIndex) => {
    addWidget(node, row, index, nodeData, values?.[rowIndex]);
    if (rowIndex === 0) {
      addRowRemoveButton(node, config, index);
    }
  });
  moveButtonToEnd(node);
  resizeNode(node);
}

function removeLastRow(node, config) {
  const widgets = dynamicWidgets(node, config);
  if (!widgets.length) {
    return;
  }
  let lastIndex = 1;
  for (const widget of widgets) {
    for (const row of config.rows) {
      if (!widget.name?.startsWith(row.prefix)) continue;
      const index = Number(widget.name.slice(row.prefix.length));
      if (Number.isFinite(index)) {
        lastIndex = Math.max(lastIndex, index);
      }
    }
  }
  for (const widget of [...widgets]) {
    for (const row of config.rows) {
      if (!widget.name?.startsWith(row.prefix)) continue;
      const index = Number(widget.name.slice(row.prefix.length));
      if (index === lastIndex) {
        const widgetIndex = node.widgets.indexOf(widget);
        if (widgetIndex >= 0) node.widgets.splice(widgetIndex, 1);
      }
    }
  }
  resizeNode(node);
}

function resizeNode(node) {
  if (!node.computeSize) {
    return;
  }
  const size = node.computeSize();
  node.size[0] = Math.max(node.size[0], size[0]);
  node.size[1] = Math.max(node.size[1], size[1]);
  node.setDirtyCanvas(true, true);
}

function dynamicWidgets(node, config) {
  const prefixes = config.rows.map((row) => row.prefix);
  return (node.widgets || []).filter((widget) => {
    if (widget === node.ognAddButton) {
      return false;
    }
    if (widget === node.ognRemoveButton) {
      return false;
    }
    if (widget.ognRowRemoveIndex) {
      return false;
    }
    if (widget.ognRowSpacerIndex) {
      return false;
    }
    const prefix = prefixes.find((candidate) => widget.name?.startsWith(candidate));
    if (!prefix) {
      return false;
    }
    const index = Number(widget.name.slice(prefix.length));
    return Number.isFinite(index) && index > 1;
  });
}

function saveDynamicWidgets(node, config) {
  return dynamicWidgets(node, config).map((widget) => ({
    name: widget.name,
    value: widget.value,
  }));
}

function restoreDynamicWidgets(node, config, nodeData, saved) {
  if (!Array.isArray(saved) || !saved.length) {
    return;
  }
  const rowsByIndex = new Map();
  for (const item of saved) {
    const row = config.rows.find((candidate) => item.name?.startsWith(candidate.prefix));
    if (!row) {
      continue;
    }
    const index = Number(item.name.slice(row.prefix.length));
    if (!Number.isFinite(index) || index <= 1) {
      continue;
    }
    const values = rowsByIndex.get(index) ?? [];
    values[config.rows.indexOf(row)] = item.value;
    rowsByIndex.set(index, values);
  }
  for (const [index, values] of [...rowsByIndex.entries()].sort((a, b) => a[0] - b[0])) {
    addRowSpacer(node, config, index);
    config.rows.forEach((row, rowIndex) => {
      addWidget(node, row, index, nodeData, values?.[rowIndex]);
      if (rowIndex === 0) {
        addRowRemoveButton(node, config, index);
      }
    });
  }
  moveButtonToEnd(node);
  resizeNode(node);
}

function parseLoraWidgetName(name) {
  let match = /^lora_(\d+)$/.exec(name ?? "");
  if (match) return { kind: "lora", set: Number(match[1]), item: 1 };
  match = /^strength_model_(\d+)$/.exec(name ?? "");
  if (match) return { kind: "strength", set: Number(match[1]), item: 1 };
  match = /^lora(\d+)_(\d+)$/.exec(name ?? "");
  if (match) return { kind: "lora", set: Number(match[1]), item: Number(match[2]) };
  match = /^strength_model_(\d+)_(\d+)$/.exec(name ?? "");
  if (match) return { kind: "strength", set: Number(match[1]), item: Number(match[2]) };
  return null;
}

function loraWidgetName(kind, set, item) {
  if (item === 1) {
    return kind === "lora" ? `lora_${set}` : `strength_model_${set}`;
  }
  return kind === "lora" ? `lora${set}_${item}` : `strength_model_${set}_${item}`;
}

function loraSetIndexes(node) {
  const indexes = new Set([1]);
  for (const widget of node.widgets || []) {
    const parsed = parseLoraWidgetName(widget.name);
    if (parsed) indexes.add(parsed.set);
  }
  return [...indexes].sort((a, b) => a - b);
}

function nextLoraSet(node) {
  return Math.max(...loraSetIndexes(node)) + 1;
}

function nextLoraItem(node, set) {
  let item = 1;
  for (const widget of node.widgets || []) {
    const parsed = parseLoraWidgetName(widget.name);
    if (parsed?.set === set) item = Math.max(item, parsed.item);
  }
  return item + 1;
}

function addCustomWidgetBeforeLoraButtons(node, widget) {
  const buttonIndexes = [node.ognAddLoraButton, node.ognAddSetButton, node.ognButtonSpacer]
    .map((button) => node.widgets?.indexOf(button) ?? -1)
    .filter((index) => index >= 0);
  const index = buttonIndexes.length ? Math.min(...buttonIndexes) : (node.widgets?.length ?? 0);
  node.widgets.splice(index, 0, widget);
}

function ensureLoraSetHeader(node, set) {
  if ((node.widgets || []).some((widget) => widget.ognLoraHeaderSet === set)) return;
  const widget = {
    type: "custom",
    name: `ogn_lora_set_${set}`,
    value: null,
    serialize: false,
    ognLoraHeaderSet: set,
    computeSize: () => [0, 22],
    draw: (ctx, _node, _width, y) => {
      ctx.save();
      ctx.fillStyle = "#ddd";
      ctx.font = "12px sans-serif";
      ctx.fillText(`[Set ${set}]`, 10, y + 15);
      ctx.restore();
    },
  };
  if (set === 1) {
    const firstWidgetIndex = node.widgets.findIndex((candidate) => candidate.name === "lora_1");
    node.widgets.splice(Math.max(firstWidgetIndex, 0), 0, widget);
  } else {
    addCustomWidgetBeforeLoraButtons(node, widget);
  }
}

function ensureLoraSetSpacer(node, set) {
  if (set <= 1 || (node.widgets || []).some((widget) => widget.ognLoraSetSpacer === set)) return;
  addCustomWidgetBeforeLoraButtons(node, {
    type: "custom",
    name: `ogn_lora_set_spacer_${set}`,
    value: null,
    serialize: false,
    ognLoraSetSpacer: set,
    computeSize: () => [0, 10],
    draw: () => {},
  });
}

function addLoraValueWidget(node, nodeData, set, item, kind, value = null) {
  const name = loraWidgetName(kind, set, item);
  if ((node.widgets || []).some((widget) => widget.name === name)) return;
  if (kind === "lora") {
    const options = getOptions(nodeData, "lora_1");
    const selected = options.includes(value) ? value : (value ?? options[0] ?? "None");
    node.addWidget("combo", name, selected, null, { values: options });
    return;
  }
  node.addWidget("number", name, value ?? 1.0, null, {
    min: -20,
    max: 20,
    step: 0.05,
    precision: 3,
  });
}

function loraItemCount(node, set) {
  const items = new Set();
  for (const widget of node.widgets || []) {
    const parsed = parseLoraWidgetName(widget.name);
    if (parsed?.set === set) items.add(parsed.item);
  }
  return items.size;
}

function removeLoraItem(node, set, item) {
  if (set === 1 && item === 1) {
    for (const widget of node.widgets || []) {
      if (widget.name === "lora_1") widget.value = "None";
      if (widget.name === "strength_model_1") widget.value = 0;
    }
    resizeNode(node);
    return;
  }
  for (const widget of [...(node.widgets || [])]) {
    const parsed = parseLoraWidgetName(widget.name);
    const shouldRemove =
      (parsed?.set === set && parsed.item === item) ||
      (widget.ognLoraItemRemove?.set === set && widget.ognLoraItemRemove?.item === item);
    if (!shouldRemove) continue;
    const index = node.widgets.indexOf(widget);
    if (index >= 0) node.widgets.splice(index, 1);
  }
  if (set > 1 && loraItemCount(node, set) === 0) {
    for (const widget of [...(node.widgets || [])]) {
      if (widget.ognLoraHeaderSet !== set && widget.ognLoraSetSpacer !== set) continue;
      const index = node.widgets.indexOf(widget);
      if (index >= 0) node.widgets.splice(index, 1);
    }
  }
  resizeNode(node);
}

function addLoraItemRemoveButton(node, set, item) {
  for (const widget of [...(node.widgets || [])]) {
    if (widget.ognLoraItemRemove?.set !== set || widget.ognLoraItemRemove?.item !== item) continue;
    const index = node.widgets.indexOf(widget);
    if (index >= 0) node.widgets.splice(index, 1);
  }
  const widget = node.addWidget("button", "- Remove", null, () => removeLoraItem(node, set, item));
  widget.ognLoraItemRemove = { set, item };
}

function addLoraToSet(node, nodeData, set, item, values = {}) {
  ensureLoraSetSpacer(node, set);
  ensureLoraSetHeader(node, set);
  addLoraValueWidget(node, nodeData, set, item, "lora", values.lora);
  addLoraValueWidget(node, nodeData, set, item, "strength", values.strength);
  addLoraItemRemoveButton(node, set, item);
}

function moveLoraButtonsToEnd(node) {
  if (!node.widgets) return;
  if ((node.ognAddLoraButton || node.ognAddSetButton) && !node.ognButtonSpacer) {
    node.ognButtonSpacer = {
      type: "custom",
      name: "ogn_button_spacer",
      value: null,
      serialize: false,
      computeSize: () => [0, 10],
      draw: () => {},
    };
  }
  for (const widget of [node.ognButtonSpacer, node.ognAddLoraButton, node.ognAddSetButton]) {
    if (!widget) continue;
    const index = node.widgets.indexOf(widget);
    if (index >= 0) node.widgets.splice(index, 1);
    node.widgets.push(widget);
  }
}

function ensureLoraButtons(node, nodeData) {
  node.serialize_widgets = true;
  ensureLoraSetHeader(node, 1);
  if (!node.ognAddLoraButton) {
    node.ognAddLoraButton = node.addWidget("button", "+ Add LoRA", null, () => {
      const set = Math.max(...loraSetIndexes(node));
      addLoraToSet(node, nodeData, set, nextLoraItem(node, set));
      moveLoraButtonsToEnd(node);
      resizeNode(node);
    });
  }
  if (!node.ognAddSetButton) {
    node.ognAddSetButton = node.addWidget("button", "+ Add Set", null, () => {
      const set = nextLoraSet(node);
      addLoraToSet(node, nodeData, set, 1);
      moveLoraButtonsToEnd(node);
      resizeNode(node);
    });
  }
  for (const set of loraSetIndexes(node)) {
    ensureLoraSetHeader(node, set);
    if (set > 1) ensureLoraSetSpacer(node, set);
    const items = new Set();
    for (const widget of node.widgets || []) {
      const parsed = parseLoraWidgetName(widget.name);
      if (parsed?.set === set) items.add(parsed.item);
    }
    for (const item of [...items].sort((a, b) => a - b)) {
      addLoraItemRemoveButton(node, set, item);
    }
  }
  moveLoraButtonsToEnd(node);
}

function saveLoraDynamicWidgets(node) {
  return (node.widgets || [])
    .filter((widget) => {
      const parsed = parseLoraWidgetName(widget.name);
      return parsed && !(parsed.set === 1 && parsed.item === 1);
    })
    .map((widget) => ({ name: widget.name, value: widget.value }));
}

function restoreLoraDynamicWidgets(node, nodeData, saved) {
  if (!Array.isArray(saved) || !saved.length) return;
  const values = new Map();
  for (const item of saved) {
    const parsed = parseLoraWidgetName(item.name);
    if (!parsed || (parsed.set === 1 && parsed.item === 1)) continue;
    const key = `${parsed.set}:${parsed.item}`;
    const row = values.get(key) ?? { set: parsed.set, item: parsed.item };
    row[parsed.kind] = item.value;
    values.set(key, row);
  }
  for (const row of [...values.values()].sort((a, b) => a.set - b.set || a.item - b.item)) {
    addLoraToSet(node, nodeData, row.set, row.item, row);
  }
  moveLoraButtonsToEnd(node);
  resizeNode(node);
}

function ensureButton(node, config, nodeData) {
  node.serialize_widgets = true;
  if (node.ognAddButton && (node.ognRemoveButton || !config.removeButton)) {
    moveButtonToEnd(node);
    return;
  }
  if (!node.ognAddButton) {
    node.ognAddButton = node.addWidget("button", config.button, null, () => {
      addRow(node, config, nodeData);
    });
  }
  if (config.removeButton && !node.ognRemoveButton) {
    node.ognRemoveButton = node.addWidget("button", config.removeButton, null, () => {
      removeLastRow(node, config);
    });
  }
  moveButtonToEnd(node);
}

app.registerExtension({
  name: "ogn.xy_plot.dynamic_axes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const config = NODE_CONFIG[nodeData.name];
    if (!config) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      if (config.loraSets) {
        ensureLoraButtons(this, nodeData);
        return;
      }
      ensureButton(this, config, nodeData);
    };

    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      configure?.apply(this, arguments);
      if (config.loraSets) {
        ensureLoraButtons(this, nodeData);
        restoreLoraDynamicWidgets(this, nodeData, info?.ogn_dynamic_widgets);
        return;
      }
      ensureButton(this, config, nodeData);
      restoreDynamicWidgets(this, config, nodeData, info?.ogn_dynamic_widgets);
    };

    const onSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (data) {
      onSerialize?.apply(this, arguments);
      if (config.loraSets) {
        data.ogn_dynamic_widgets = saveLoraDynamicWidgets(this);
        return;
      }
      data.ogn_dynamic_widgets = saveDynamicWidgets(this, config);
    };
  },
});
