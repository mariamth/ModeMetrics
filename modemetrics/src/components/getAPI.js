import axios from "axios";

const getApiData = async () => {
    const apiKey = 'e00876c257msh069b3f05a5c7468p100529jsn449ab194ca84';
    const baseUrl = "https://asos2.p.rapidapi.com/products/v2/list";
    const backendUrl = "http://127.0.0.1:8000/api/store_data/";
    console.log(process.env);

    console.log("API Key:", process.env.ASOS_API_KEY);




    const options = {
        method: 'GET',
        url: baseUrl,
        params: {
            store: 'COM',
            offset: '0',
            categoryId: '17184',
            country: 'GB',
            sort: 'freshness',
            currency: 'GBP',
            sizeSchema: 'EU',
            limit: '48', 
            lang: 'en-GB',
        },
        headers: {
            'x-rapidapi-key': apiKey,
            'x-rapidapi-host': 'asos2.p.rapidapi.com'
        }
        
    };

    try {
        
        const response = await axios.request(options);

        const formattedProducts = response.data.products.map(product => ({
            id: product.id.toString(),
            name: product.name,
            price: product.price.current.value,
            is_new: product.isNew || false,
            
        }));
          

        const backendResponse = await axios.post(
            backendUrl,
            { products: formattedProducts },
            { headers: { 'Content-Type': 'application/json' } }
        );

        console.log("Backend response:", backendResponse.data);
        return formattedProducts;

    } catch (error) {
        console.error("API Error:", error.message);
        return []; 
    }
};

export default getApiData;
