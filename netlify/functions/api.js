exports.handler = async (event, context) => {
  const response = {
    composite: [6, 12, 19, 24, 31, 33],
    lstm: [6, 12, 19, 24, 31, 38],
    rf: [5, 6, 12, 29, 33, 44],
    reg: [1, 7, 15, 26, 33, 45]
  };
  return {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(response)
  };
};
