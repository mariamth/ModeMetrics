import axios from "axios";

const getApiData = async () => {
    const apiKey = 'e00876c257msh069b3f05a5c7468p100529jsn449ab194ca84';
    // const apiKey = process.env.ASOS_API_KEY;
    const baseUrl = "https://asos2.p.rapidapi.com/products/v2/list";
    console.log(process.env);

    console.log("API Key:", process.env.ASOS_API_KEY);



    const options = {
        method: 'GET',
        url: baseUrl,
        params: {
            store: 'US',
            offset: '0',
            categoryId: '4209',
            country: 'US',
            sort: 'freshness',
            currency: 'USD',
            sizeSchema: 'US',
            limit: '10', // Reduce to avoid hitting rate limits
            lang: 'en-US'
        },
        headers: {
            'x-rapidapi-key': apiKey,
            'x-rapidapi-host': 'asos2.p.rapidapi.com'
        }
    };

    try {
        const response = await axios.request(options);

        if (response.status === 401) {
            throw new Error("Unauthorized: Check your API key.");
        }

        if (response.status === 429) {
            throw new Error("Too Many Requests: Slow down or upgrade your plan.");
        }

        console.log("API Response:", response.data);
        return response.data.products || [];  // ✅ Always return an array

    } catch (error) {
        console.error("API Error:", error.message);
        return [];  // ✅ Prevents app crash
    }
};

export default getApiData;
